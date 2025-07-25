"""
Digital Storefront Management Models

Database models for managing e-commerce storefronts, themes, content, and layout.
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, DateTime, Numeric, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum

Base = declarative_base()

class StorefrontStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	INACTIVE = "inactive"
	MAINTENANCE = "maintenance"

class ThemeType(str, Enum):
	RETAIL = "retail"
	MARKETPLACE = "marketplace"
	BUSINESS = "business"
	CUSTOM = "custom"

class WidgetType(str, Enum):
	BANNER = "banner"
	PRODUCT_GRID = "product_grid"
	CAROUSEL = "carousel"
	TEXT_BLOCK = "text_block"
	VIDEO = "video"
	TESTIMONIALS = "testimonials"
	NEWSLETTER = "newsletter"
	SOCIAL_FEED = "social_feed"

class PageType(str, Enum):
	HOME = "home"
	CATEGORY = "category"
	PRODUCT = "product"
	ABOUT = "about"
	CONTACT = "contact"
	TERMS = "terms"
	PRIVACY = "privacy"
	CUSTOM = "custom"

# SQLAlchemy Models
class PSStorefront(Base):
	__tablename__ = 'ps_storefronts'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Basic Information
	name = Column(String(255), nullable=False)
	code = Column(String(50), nullable=False, unique=True)
	domain = Column(String(255), nullable=True)
	subdomain = Column(String(100), nullable=True)
	description = Column(Text)
	
	# Status and Configuration
	status = Column(String(20), nullable=False, default=StorefrontStatus.DRAFT.value)
	is_primary = Column(Boolean, default=False)
	is_mobile_optimized = Column(Boolean, default=True)
	is_pwa_enabled = Column(Boolean, default=False)
	
	# Branding
	logo_url = Column(String(500))
	favicon_url = Column(String(500))
	brand_colors = Column(JSON)  # {"primary": "#...", "secondary": "#..."}
	brand_fonts = Column(JSON)   # {"heading": "...", "body": "..."}
	
	# SEO and Meta
	meta_title = Column(String(255))
	meta_description = Column(Text)
	meta_keywords = Column(Text)
	og_image_url = Column(String(500))
	
	# Configuration
	theme_id = Column(String(36), ForeignKey('ps_storefront_themes.id'))
	default_language = Column(String(10), default='en')
	default_currency = Column(String(3), default='USD')
	timezone = Column(String(50), default='UTC')
	
	# Features
	features = Column(JSON)  # {"search": true, "reviews": true, "chat": false}
	integrations = Column(JSON)  # {"analytics": "...", "payment": "..."}
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	theme = relationship("PSStorefrontTheme", back_populates="storefronts")
	pages = relationship("PSStorefrontPage", back_populates="storefront", cascade="all, delete-orphan")
	navigation = relationship("PSStorefrontNavigation", back_populates="storefront", cascade="all, delete-orphan")
	banners = relationship("PSStorefrontBanner", back_populates="storefront", cascade="all, delete-orphan")
	layouts = relationship("PSStorefrontLayout", back_populates="storefront", cascade="all, delete-orphan")
	seo_settings = relationship("PSStorefrontSEO", back_populates="storefront", cascade="all, delete-orphan")

class PSStorefrontTheme(Base):
	__tablename__ = 'ps_storefront_themes'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	
	# Theme Information
	name = Column(String(255), nullable=False)
	code = Column(String(50), nullable=False)
	description = Column(Text)
	version = Column(String(20), default='1.0.0')
	
	# Theme Type and Configuration
	theme_type = Column(String(20), nullable=False, default=ThemeType.RETAIL.value)
	is_responsive = Column(Boolean, default=True)
	is_custom = Column(Boolean, default=False)
	is_active = Column(Boolean, default=True)
	
	# Assets
	preview_image_url = Column(String(500))
	css_files = Column(JSON)  # List of CSS file URLs
	js_files = Column(JSON)   # List of JS file URLs
	template_files = Column(JSON)  # Template file mappings
	
	# Customization Options
	color_scheme = Column(JSON)  # Available color options
	font_options = Column(JSON)  # Available font options
	layout_options = Column(JSON)  # Layout configuration options
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	storefronts = relationship("PSStorefront", back_populates="theme")

class PSStorefrontPage(Base):
	__tablename__ = 'ps_storefront_pages'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	storefront_id = Column(String(36), ForeignKey('ps_storefronts.id'), nullable=False)
	
	# Page Information
	title = Column(String(255), nullable=False)
	slug = Column(String(255), nullable=False)
	page_type = Column(String(20), nullable=False, default=PageType.CUSTOM.value)
	
	# Content
	content = Column(Text)  # HTML/Rich content
	excerpt = Column(Text)
	featured_image_url = Column(String(500))
	
	# Configuration
	is_published = Column(Boolean, default=False)
	is_featured = Column(Boolean, default=False)
	template = Column(String(100))  # Template file to use
	layout_id = Column(String(36), ForeignKey('ps_storefront_layouts.id'))
	
	# SEO
	meta_title = Column(String(255))
	meta_description = Column(Text)
	meta_keywords = Column(Text)
	
	# Display Order
	sort_order = Column(Integer, default=0)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	published_at = Column(DateTime)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	storefront = relationship("PSStorefront", back_populates="pages")
	layout = relationship("PSStorefrontLayout", back_populates="pages")
	widgets = relationship("PSStorefrontWidget", back_populates="page", cascade="all, delete-orphan")

class PSStorefrontWidget(Base):
	__tablename__ = 'ps_storefront_widgets'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	page_id = Column(String(36), ForeignKey('ps_storefront_pages.id'), nullable=False)
	
	# Widget Information
	name = Column(String(255), nullable=False)
	widget_type = Column(String(50), nullable=False)
	
	# Configuration
	configuration = Column(JSON)  # Widget-specific settings
	content = Column(Text)  # Widget content (if applicable)
	
	# Position and Display
	zone = Column(String(50))  # header, sidebar, content, footer
	position = Column(Integer, default=0)
	is_active = Column(Boolean, default=True)
	is_mobile_visible = Column(Boolean, default=True)
	
	# Styling
	css_classes = Column(String(255))
	inline_styles = Column(Text)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	page = relationship("PSStorefrontPage", back_populates="widgets")

class PSStorefrontNavigation(Base):
	__tablename__ = 'ps_storefront_navigation'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	storefront_id = Column(String(36), ForeignKey('ps_storefronts.id'), nullable=False)
	
	# Navigation Information
	name = Column(String(255), nullable=False)
	location = Column(String(50), nullable=False)  # header, footer, sidebar
	
	# Menu Item
	label = Column(String(255), nullable=False)
	url = Column(String(500))
	page_id = Column(String(36), ForeignKey('ps_storefront_pages.id'), nullable=True)
	category_id = Column(String(36), nullable=True)  # Link to product categories
	
	# Hierarchy
	parent_id = Column(String(36), ForeignKey('ps_storefront_navigation.id'), nullable=True)
	sort_order = Column(Integer, default=0)
	level = Column(Integer, default=0)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_featured = Column(Boolean, default=False)
	open_in_new_tab = Column(Boolean, default=False)
	css_class = Column(String(100))
	icon = Column(String(100))
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	storefront = relationship("PSStorefront", back_populates="navigation")
	parent = relationship("PSStorefrontNavigation", remote_side=[id])
	page = relationship("PSStorefrontPage")

class PSStorefrontBanner(Base):
	__tablename__ = 'ps_storefront_banners'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	storefront_id = Column(String(36), ForeignKey('ps_storefronts.id'), nullable=False)
	
	# Banner Information
	title = Column(String(255), nullable=False)
	subtitle = Column(Text)
	description = Column(Text)
	
	# Media
	image_url = Column(String(500))
	mobile_image_url = Column(String(500))
	video_url = Column(String(500))
	
	# Call to Action
	cta_text = Column(String(100))
	cta_url = Column(String(500))
	cta_style = Column(String(50))  # button, link, none
	
	# Display Configuration
	location = Column(String(50), nullable=False)  # hero, sidebar, footer
	display_type = Column(String(50), default='image')  # image, video, slider
	sort_order = Column(Integer, default=0)
	
	# Scheduling
	start_date = Column(DateTime)
	end_date = Column(DateTime)
	is_active = Column(Boolean, default=True)
	
	# Styling
	background_color = Column(String(20))
	text_color = Column(String(20))
	overlay_opacity = Column(Numeric(3, 2), default=0.0)
	
	# Analytics
	click_count = Column(Integer, default=0)
	impression_count = Column(Integer, default=0)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	storefront = relationship("PSStorefront", back_populates="banners")

class PSStorefrontLayout(Base):
	__tablename__ = 'ps_storefront_layouts'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	storefront_id = Column(String(36), ForeignKey('ps_storefronts.id'), nullable=False)
	
	# Layout Information
	name = Column(String(255), nullable=False)
	code = Column(String(50), nullable=False)
	description = Column(Text)
	
	# Layout Configuration
	template_file = Column(String(255))
	css_file = Column(String(255))
	js_file = Column(String(255))
	
	# Grid Configuration
	grid_columns = Column(Integer, default=12)
	grid_gutter = Column(String(20), default='20px')
	max_width = Column(String(20), default='1200px')
	
	# Responsive Breakpoints
	breakpoints = Column(JSON)  # {"sm": 576, "md": 768, "lg": 992, "xl": 1200}
	
	# Zone Configuration
	zones = Column(JSON)  # Available content zones and their properties
	
	# Status
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	storefront = relationship("PSStorefront", back_populates="layouts")
	pages = relationship("PSStorefrontPage", back_populates="layout")

class PSStorefrontSEO(Base):
	__tablename__ = 'ps_storefront_seo'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False)
	storefront_id = Column(String(36), ForeignKey('ps_storefronts.id'), nullable=False)
	
	# SEO Configuration
	robots_txt = Column(Text)
	sitemap_url = Column(String(500))
	google_analytics_id = Column(String(50))
	google_tag_manager_id = Column(String(50))
	facebook_pixel_id = Column(String(50))
	
	# Schema Markup
	organization_schema = Column(JSON)
	website_schema = Column(JSON)
	breadcrumb_schema = Column(JSON)
	
	# Social Media
	og_site_name = Column(String(255))
	twitter_handle = Column(String(100))
	facebook_app_id = Column(String(50))
	
	# Performance
	enable_caching = Column(Boolean, default=True)
	cache_ttl = Column(Integer, default=3600)
	enable_compression = Column(Boolean, default=True)
	enable_cdn = Column(Boolean, default=False)
	cdn_url = Column(String(500))
	
	# Security
	ssl_enabled = Column(Boolean, default=True)
	hsts_enabled = Column(Boolean, default=False)
	csp_policy = Column(Text)
	
	# Audit fields
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36))
	updated_by = Column(String(36))
	
	# Relationships
	storefront = relationship("PSStorefront", back_populates="seo_settings")

# Pydantic Models for API/Views
class StorefrontCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	code: str = Field(..., min_length=1, max_length=50)
	domain: str | None = Field(None, max_length=255)
	subdomain: str | None = Field(None, max_length=100)
	description: str | None = None
	theme_id: str | None = None
	default_language: str = Field(default='en', max_length=10)
	default_currency: str = Field(default='USD', max_length=3)
	timezone: str = Field(default='UTC', max_length=50)

class StorefrontUpdate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str | None = Field(None, min_length=1, max_length=255)
	domain: str | None = Field(None, max_length=255)
	subdomain: str | None = Field(None, max_length=100)
	description: str | None = None
	status: StorefrontStatus | None = None
	theme_id: str | None = None
	meta_title: str | None = Field(None, max_length=255)
	meta_description: str | None = None
	brand_colors: Dict[str, str] | None = None
	features: Dict[str, Any] | None = None

class PageCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	title: str = Field(..., min_length=1, max_length=255)
	slug: str = Field(..., min_length=1, max_length=255)
	page_type: PageType = Field(default=PageType.CUSTOM)
	content: str | None = None
	template: str | None = Field(None, max_length=100)
	layout_id: str | None = None

class BannerCreate(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	title: str = Field(..., min_length=1, max_length=255)
	location: str = Field(..., min_length=1, max_length=50)
	image_url: str | None = Field(None, max_length=500)
	cta_text: str | None = Field(None, max_length=100)
	cta_url: str | None = Field(None, max_length=500)
	start_date: datetime | None = None
	end_date: datetime | None = None