"""
Digital Storefront Management Service

Business logic for managing e-commerce storefronts, themes, content, and layout.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, timedelta
from decimal import Decimal
import json
import logging

from .models import (
	PSStorefront, PSStorefrontTheme, PSStorefrontPage, PSStorefrontWidget,
	PSStorefrontNavigation, PSStorefrontBanner, PSStorefrontLayout, PSStorefrontSEO,
	StorefrontStatus, ThemeType, PageType, WidgetType,
	StorefrontCreate, StorefrontUpdate, PageCreate, BannerCreate
)

logger = logging.getLogger(__name__)

class DigitalStorefrontService:
	"""Service class for digital storefront management operations"""
	
	def __init__(self, db_session: Session):
		self.db = db_session
	
	# Storefront Management
	def create_storefront(self, tenant_id: str, storefront_data: StorefrontCreate, created_by: str) -> PSStorefront:
		"""Create a new storefront"""
		# Check if code is unique
		existing = self.db.query(PSStorefront).filter(
			and_(
				PSStorefront.tenant_id == tenant_id,
				PSStorefront.code == storefront_data.code
			)
		).first()
		
		if existing:
			raise ValueError(f"Storefront with code '{storefront_data.code}' already exists")
		
		storefront = PSStorefront(
			tenant_id=tenant_id,
			**storefront_data.model_dump(exclude_none=True),
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(storefront)
		self.db.commit()
		self.db.refresh(storefront)
		
		# Create default SEO settings
		self._create_default_seo_settings(storefront.id, tenant_id, created_by)
		
		# Create default layout
		self._create_default_layout(storefront.id, tenant_id, created_by)
		
		self._log_storefront_created(storefront)
		return storefront
	
	def get_storefront(self, tenant_id: str, storefront_id: str) -> Optional[PSStorefront]:
		"""Get storefront by ID"""
		return self.db.query(PSStorefront).filter(
			and_(
				PSStorefront.tenant_id == tenant_id,
				PSStorefront.id == storefront_id
			)
		).first()
	
	def get_storefront_by_code(self, tenant_id: str, code: str) -> Optional[PSStorefront]:
		"""Get storefront by code"""
		return self.db.query(PSStorefront).filter(
			and_(
				PSStorefront.tenant_id == tenant_id,
				PSStorefront.code == code
			)
		).first()
	
	def get_storefront_by_domain(self, domain: str) -> Optional[PSStorefront]:
		"""Get storefront by domain"""
		return self.db.query(PSStorefront).filter(
			or_(
				PSStorefront.domain == domain,
				PSStorefront.subdomain == domain.split('.')[0]
			)
		).first()
	
	def update_storefront(self, tenant_id: str, storefront_id: str, update_data: StorefrontUpdate, updated_by: str) -> Optional[PSStorefront]:
		"""Update storefront"""
		storefront = self.get_storefront(tenant_id, storefront_id)
		if not storefront:
			return None
		
		update_dict = update_data.model_dump(exclude_none=True)
		for key, value in update_dict.items():
			setattr(storefront, key, value)
		
		storefront.updated_by = updated_by
		storefront.updated_at = datetime.utcnow()
		
		self.db.commit()
		self.db.refresh(storefront)
		
		self._log_storefront_updated(storefront)
		return storefront
	
	def list_storefronts(self, tenant_id: str, status: Optional[StorefrontStatus] = None) -> List[PSStorefront]:
		"""List storefronts with optional status filter"""
		query = self.db.query(PSStorefront).filter(PSStorefront.tenant_id == tenant_id)
		
		if status:
			query = query.filter(PSStorefront.status == status.value)
		
		return query.order_by(desc(PSStorefront.created_at)).all()
	
	def activate_storefront(self, tenant_id: str, storefront_id: str, updated_by: str) -> bool:
		"""Activate a storefront"""
		storefront = self.get_storefront(tenant_id, storefront_id)
		if not storefront:
			return False
		
		storefront.status = StorefrontStatus.ACTIVE.value
		storefront.updated_by = updated_by
		storefront.updated_at = datetime.utcnow()
		
		self.db.commit()
		self._log_storefront_activated(storefront)
		return True
	
	def set_primary_storefront(self, tenant_id: str, storefront_id: str, updated_by: str) -> bool:
		"""Set a storefront as primary (only one can be primary)"""
		# Unset current primary
		self.db.query(PSStorefront).filter(
			and_(
				PSStorefront.tenant_id == tenant_id,
				PSStorefront.is_primary == True
			)
		).update({"is_primary": False, "updated_by": updated_by, "updated_at": datetime.utcnow()})
		
		# Set new primary
		storefront = self.get_storefront(tenant_id, storefront_id)
		if not storefront:
			return False
		
		storefront.is_primary = True
		storefront.updated_by = updated_by
		storefront.updated_at = datetime.utcnow()
		
		self.db.commit()
		return True
	
	# Theme Management
	def create_theme(self, tenant_id: str, name: str, code: str, theme_type: ThemeType, created_by: str) -> PSStorefrontTheme:
		"""Create a new storefront theme"""
		theme = PSStorefrontTheme(
			tenant_id=tenant_id,
			name=name,
			code=code,
			theme_type=theme_type.value,
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(theme)
		self.db.commit()
		self.db.refresh(theme)
		
		self._log_theme_created(theme)
		return theme
	
	def list_themes(self, tenant_id: str, theme_type: Optional[ThemeType] = None, is_active: bool = True) -> List[PSStorefrontTheme]:
		"""List available themes"""
		query = self.db.query(PSStorefrontTheme).filter(PSStorefrontTheme.tenant_id == tenant_id)
		
		if theme_type:
			query = query.filter(PSStorefrontTheme.theme_type == theme_type.value)
		
		if is_active is not None:
			query = query.filter(PSStorefrontTheme.is_active == is_active)
		
		return query.order_by(PSStorefrontTheme.name).all()
	
	# Page Management
	def create_page(self, tenant_id: str, storefront_id: str, page_data: PageCreate, created_by: str) -> PSStorefrontPage:
		"""Create a new storefront page"""
		# Check if slug is unique within storefront
		existing = self.db.query(PSStorefrontPage).filter(
			and_(
				PSStorefrontPage.tenant_id == tenant_id,
				PSStorefrontPage.storefront_id == storefront_id,
				PSStorefrontPage.slug == page_data.slug
			)
		).first()
		
		if existing:
			raise ValueError(f"Page with slug '{page_data.slug}' already exists in this storefront")
		
		page = PSStorefrontPage(
			tenant_id=tenant_id,
			storefront_id=storefront_id,
			**page_data.model_dump(exclude_none=True),
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(page)
		self.db.commit()
		self.db.refresh(page)
		
		self._log_page_created(page)
		return page
	
	def get_page_by_slug(self, tenant_id: str, storefront_id: str, slug: str) -> Optional[PSStorefrontPage]:
		"""Get page by slug"""
		return self.db.query(PSStorefrontPage).filter(
			and_(
				PSStorefrontPage.tenant_id == tenant_id,
				PSStorefrontPage.storefront_id == storefront_id,
				PSStorefrontPage.slug == slug,
				PSStorefrontPage.is_published == True
			)
		).first()
	
	def list_pages(self, tenant_id: str, storefront_id: str, page_type: Optional[PageType] = None, published_only: bool = False) -> List[PSStorefrontPage]:
		"""List storefront pages"""
		query = self.db.query(PSStorefrontPage).filter(
			and_(
				PSStorefrontPage.tenant_id == tenant_id,
				PSStorefrontPage.storefront_id == storefront_id
			)
		)
		
		if page_type:
			query = query.filter(PSStorefrontPage.page_type == page_type.value)
		
		if published_only:
			query = query.filter(PSStorefrontPage.is_published == True)
		
		return query.order_by(PSStorefrontPage.sort_order, PSStorefrontPage.title).all()
	
	def publish_page(self, tenant_id: str, page_id: str, updated_by: str) -> bool:
		"""Publish a page"""
		page = self.db.query(PSStorefrontPage).filter(
			and_(
				PSStorefrontPage.tenant_id == tenant_id,
				PSStorefrontPage.id == page_id
			)
		).first()
		
		if not page:
			return False
		
		page.is_published = True
		page.published_at = datetime.utcnow()
		page.updated_by = updated_by
		page.updated_at = datetime.utcnow()
		
		self.db.commit()
		return True
	
	# Banner Management
	def create_banner(self, tenant_id: str, storefront_id: str, banner_data: BannerCreate, created_by: str) -> PSStorefrontBanner:
		"""Create a new storefront banner"""
		banner = PSStorefrontBanner(
			tenant_id=tenant_id,
			storefront_id=storefront_id,
			**banner_data.model_dump(exclude_none=True),
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(banner)
		self.db.commit()
		self.db.refresh(banner)
		
		self._log_banner_created(banner)
		return banner
	
	def get_active_banners(self, tenant_id: str, storefront_id: str, location: Optional[str] = None) -> List[PSStorefrontBanner]:
		"""Get active banners for a storefront location"""
		now = datetime.utcnow()
		query = self.db.query(PSStorefrontBanner).filter(
			and_(
				PSStorefrontBanner.tenant_id == tenant_id,
				PSStorefrontBanner.storefront_id == storefront_id,
				PSStorefrontBanner.is_active == True,
				or_(
					PSStorefrontBanner.start_date.is_(None),
					PSStorefrontBanner.start_date <= now
				),
				or_(
					PSStorefrontBanner.end_date.is_(None),
					PSStorefrontBanner.end_date >= now
				)
			)
		)
		
		if location:
			query = query.filter(PSStorefrontBanner.location == location)
		
		return query.order_by(PSStorefrontBanner.sort_order).all()
	
	def track_banner_impression(self, banner_id: str) -> bool:
		"""Track banner impression"""
		banner = self.db.query(PSStorefrontBanner).filter(PSStorefrontBanner.id == banner_id).first()
		if banner:
			banner.impression_count += 1
			self.db.commit()
			return True
		return False
	
	def track_banner_click(self, banner_id: str) -> bool:
		"""Track banner click"""
		banner = self.db.query(PSStorefrontBanner).filter(PSStorefrontBanner.id == banner_id).first()
		if banner:
			banner.click_count += 1
			self.db.commit()
			return True
		return False
	
	# Navigation Management
	def create_navigation_item(self, tenant_id: str, storefront_id: str, label: str, location: str, 
							 url: Optional[str] = None, page_id: Optional[str] = None, 
							 parent_id: Optional[str] = None, created_by: str = None) -> PSStorefrontNavigation:
		"""Create navigation menu item"""
		nav_item = PSStorefrontNavigation(
			tenant_id=tenant_id,
			storefront_id=storefront_id,
			name=label,
			label=label,
			location=location,
			url=url,
			page_id=page_id,
			parent_id=parent_id,
			created_by=created_by,
			updated_by=created_by
		)
		
		# Set level based on parent
		if parent_id:
			parent = self.db.query(PSStorefrontNavigation).filter(PSStorefrontNavigation.id == parent_id).first()
			if parent:
				nav_item.level = parent.level + 1
		
		self.db.add(nav_item)
		self.db.commit()
		self.db.refresh(nav_item)
		
		return nav_item
	
	def get_navigation_menu(self, tenant_id: str, storefront_id: str, location: str) -> List[PSStorefrontNavigation]:
		"""Get navigation menu for a location"""
		return self.db.query(PSStorefrontNavigation).filter(
			and_(
				PSStorefrontNavigation.tenant_id == tenant_id,
				PSStorefrontNavigation.storefront_id == storefront_id,
				PSStorefrontNavigation.location == location,
				PSStorefrontNavigation.is_active == True
			)
		).order_by(PSStorefrontNavigation.sort_order, PSStorefrontNavigation.label).all()
	
	# Analytics and Reporting
	def get_storefront_analytics(self, tenant_id: str, storefront_id: str, days: int = 30) -> Dict[str, Any]:
		"""Get storefront analytics"""
		since_date = datetime.utcnow() - timedelta(days=days)
		
		# Banner performance
		banner_stats = self.db.query(PSStorefrontBanner).filter(
			and_(
				PSStorefrontBanner.tenant_id == tenant_id,
				PSStorefrontBanner.storefront_id == storefront_id,
				PSStorefrontBanner.created_at >= since_date
			)
		).all()
		
		total_impressions = sum(b.impression_count for b in banner_stats)
		total_clicks = sum(b.click_count for b in banner_stats)
		ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
		
		# Page counts
		total_pages = self.db.query(PSStorefrontPage).filter(
			and_(
				PSStorefrontPage.tenant_id == tenant_id,
				PSStorefrontPage.storefront_id == storefront_id
			)
		).count()
		
		published_pages = self.db.query(PSStorefrontPage).filter(
			and_(
				PSStorefrontPage.tenant_id == tenant_id,
				PSStorefrontPage.storefront_id == storefront_id,
				PSStorefrontPage.is_published == True
			)
		).count()
		
		return {
			'banner_performance': {
				'total_impressions': total_impressions,
				'total_clicks': total_clicks,
				'click_through_rate': round(ctr, 2)
			},
			'content_stats': {
				'total_pages': total_pages,
				'published_pages': published_pages,
				'draft_pages': total_pages - published_pages
			},
			'storefront_status': {
				'is_active': self.get_storefront(tenant_id, storefront_id).status == StorefrontStatus.ACTIVE.value
			}
		}
	
	# Private helper methods
	def _create_default_seo_settings(self, storefront_id: str, tenant_id: str, created_by: str):
		"""Create default SEO settings for a new storefront"""
		seo_settings = PSStorefrontSEO(
			tenant_id=tenant_id,
			storefront_id=storefront_id,
			robots_txt="User-agent: *\nDisallow:",
			enable_caching=True,
			cache_ttl=3600,
			enable_compression=True,
			ssl_enabled=True,
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(seo_settings)
		self.db.commit()
	
	def _create_default_layout(self, storefront_id: str, tenant_id: str, created_by: str):
		"""Create default layout for a new storefront"""
		layout = PSStorefrontLayout(
			tenant_id=tenant_id,
			storefront_id=storefront_id,
			name="Default Layout",
			code="default",
			description="Default storefront layout",
			grid_columns=12,
			grid_gutter="20px",
			max_width="1200px",
			breakpoints={"sm": 576, "md": 768, "lg": 992, "xl": 1200},
			zones={"header": {}, "content": {}, "sidebar": {}, "footer": {}},
			is_active=True,
			is_default=True,
			created_by=created_by,
			updated_by=created_by
		)
		
		self.db.add(layout)
		self.db.commit()
	
	# Logging methods
	def _log_storefront_created(self, storefront: PSStorefront):
		"""Log storefront creation"""
		logger.info(f"Storefront created: {storefront.name} ({storefront.code}) for tenant {storefront.tenant_id}")
	
	def _log_storefront_updated(self, storefront: PSStorefront):
		"""Log storefront update"""
		logger.info(f"Storefront updated: {storefront.name} ({storefront.id})")
	
	def _log_storefront_activated(self, storefront: PSStorefront):
		"""Log storefront activation"""
		logger.info(f"Storefront activated: {storefront.name} ({storefront.id})")
	
	def _log_theme_created(self, theme: PSStorefrontTheme):
		"""Log theme creation"""
		logger.info(f"Theme created: {theme.name} ({theme.code}) for tenant {theme.tenant_id}")
	
	def _log_page_created(self, page: PSStorefrontPage):
		"""Log page creation"""
		logger.info(f"Page created: {page.title} ({page.slug}) in storefront {page.storefront_id}")
	
	def _log_banner_created(self, banner: PSStorefrontBanner):
		"""Log banner creation"""
		logger.info(f"Banner created: {banner.title} in storefront {banner.storefront_id}")