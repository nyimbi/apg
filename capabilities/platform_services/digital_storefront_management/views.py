"""
Digital Storefront Management Views

Flask-AppBuilder views for managing e-commerce storefronts, themes, and content.
"""

from flask import request, flash, redirect, url_for, jsonify
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from flask_appbuilder.widgets import ListThumbnail
from wtforms import Form, StringField, TextAreaField, BooleanField, SelectField
from wtforms.validators import DataRequired, Length, URL, Optional
from markupsafe import Markup
import json

from .models import (
	PSStorefront, PSStorefrontTheme, PSStorefrontPage, PSStorefrontWidget,
	PSStorefrontNavigation, PSStorefrontBanner, PSStorefrontLayout, PSStorefrontSEO,
	StorefrontStatus, ThemeType, PageType
)
from .service import DigitalStorefrontService

class StorefrontModelView(ModelView):
	"""Storefront management view"""
	datamodel = SQLAInterface(PSStorefront)
	
	list_title = "Digital Storefronts"
	show_title = "Storefront Details"
	add_title = "Create Storefront"
	edit_title = "Edit Storefront"
	
	list_columns = ['name', 'code', 'domain', 'status', 'is_primary', 'theme.name', 'created_at']
	show_columns = [
		'name', 'code', 'description', 'domain', 'subdomain', 'status', 
		'is_primary', 'is_mobile_optimized', 'is_pwa_enabled',
		'theme', 'default_language', 'default_currency', 'timezone',
		'meta_title', 'meta_description', 'meta_keywords',
		'created_at', 'updated_at', 'created_by', 'updated_by'
	]
	add_columns = [
		'name', 'code', 'description', 'domain', 'subdomain',
		'theme', 'default_language', 'default_currency', 'timezone',
		'meta_title', 'meta_description', 'meta_keywords'
	]
	edit_columns = [
		'name', 'description', 'domain', 'subdomain', 'status',
		'is_primary', 'is_mobile_optimized', 'is_pwa_enabled',
		'theme', 'default_language', 'default_currency', 'timezone',
		'meta_title', 'meta_description', 'meta_keywords'
	]
	
	search_columns = ['name', 'code', 'domain', 'subdomain']
	
	base_order = ('created_at', 'desc')
	
	# Custom formatting
	formatters_columns = {
		'status': lambda x: f'<span class="badge badge-{"success" if x.status == "active" else "warning"}">{x.status.title()}</span>',
		'is_primary': lambda x: '<i class="fa fa-check text-success"></i>' if x.is_primary else '<i class="fa fa-times text-muted"></i>',
		'domain': lambda x: f'<a href="https://{x.domain}" target="_blank">{x.domain}</a>' if x.domain else '-',
		'theme.name': lambda x: x.theme.name if x.theme else '-'
	}
	
	@action("activate", "Activate", "Activate selected storefronts?", "fa-play")
	def activate_storefronts(self, items):
		"""Activate selected storefronts"""
		service = DigitalStorefrontService(self.datamodel.session)
		
		count = 0
		for item in items:
			if service.activate_storefront(item.tenant_id, item.id, "system"):
				count += 1
		
		flash(f"Activated {count} storefront(s)", "success")
		return redirect(self.get_redirect())
	
	@action("set_primary", "Set as Primary", "Set selected storefront as primary?", "fa-star")
	def set_primary_storefront(self, items):
		"""Set storefront as primary"""
		if len(items) != 1:
			flash("Please select exactly one storefront to set as primary", "warning")
			return redirect(self.get_redirect())
		
		service = DigitalStorefrontService(self.datamodel.session)
		item = items[0]
		
		if service.set_primary_storefront(item.tenant_id, item.id, "system"):
			flash(f"Set '{item.name}' as primary storefront", "success")
		else:
			flash("Failed to set primary storefront", "error")
		
		return redirect(self.get_redirect())

class StorefrontThemeModelView(ModelView):
	"""Storefront theme management view"""
	datamodel = SQLAInterface(PSStorefrontTheme)
	
	list_title = "Storefront Themes"
	show_title = "Theme Details"
	add_title = "Create Theme"
	edit_title = "Edit Theme"
	
	list_columns = ['name', 'code', 'theme_type', 'version', 'is_active', 'is_custom', 'created_at']
	show_columns = [
		'name', 'code', 'description', 'version', 'theme_type',
		'is_responsive', 'is_custom', 'is_active',
		'preview_image_url', 'css_files', 'js_files',
		'created_at', 'updated_at'
	]
	add_columns = [
		'name', 'code', 'description', 'theme_type', 'version',
		'is_responsive', 'is_custom', 'preview_image_url'
	]
	edit_columns = [
		'name', 'description', 'version', 'theme_type',
		'is_responsive', 'is_custom', 'is_active', 'preview_image_url'
	]
	
	search_columns = ['name', 'code', 'theme_type']
	
	formatters_columns = {
		'theme_type': lambda x: f'<span class="badge badge-info">{x.theme_type.title()}</span>',
		'is_active': lambda x: '<i class="fa fa-check text-success"></i>' if x.is_active else '<i class="fa fa-times text-muted"></i>',
		'is_custom': lambda x: '<i class="fa fa-check text-warning"></i>' if x.is_custom else '<i class="fa fa-times text-muted"></i>',
		'preview_image_url': lambda x: f'<img src="{x.preview_image_url}" height="50" />' if x.preview_image_url else '-'
	}

class StorefrontPageModelView(ModelView):
	"""Storefront page management view"""
	datamodel = SQLAInterface(PSStorefrontPage)
	
	list_title = "Storefront Pages"
	show_title = "Page Details"
	add_title = "Create Page"
	edit_title = "Edit Page"
	
	list_columns = ['title', 'slug', 'page_type', 'storefront.name', 'is_published', 'sort_order', 'created_at']
	show_columns = [
		'title', 'slug', 'page_type', 'content', 'excerpt',
		'storefront', 'layout', 'template', 'featured_image_url',
		'is_published', 'is_featured', 'sort_order',
		'meta_title', 'meta_description', 'meta_keywords',
		'created_at', 'updated_at', 'published_at'
	]
	add_columns = [
		'storefront', 'title', 'slug', 'page_type', 'content', 'excerpt',
		'layout', 'template', 'featured_image_url', 'sort_order',
		'meta_title', 'meta_description', 'meta_keywords'
	]
	edit_columns = [
		'title', 'slug', 'page_type', 'content', 'excerpt',
		'layout', 'template', 'featured_image_url', 'is_featured',
		'sort_order', 'meta_title', 'meta_description', 'meta_keywords'
	]
	
	search_columns = ['title', 'slug', 'content']
	
	formatters_columns = {
		'page_type': lambda x: f'<span class="badge badge-secondary">{x.page_type.title()}</span>',
		'is_published': lambda x: '<i class="fa fa-check text-success"></i>' if x.is_published else '<i class="fa fa-times text-warning"></i>',
		'is_featured': lambda x: '<i class="fa fa-star text-warning"></i>' if x.is_featured else '-',
		'storefront.name': lambda x: x.storefront.name if x.storefront else '-'
	}
	
	@action("publish", "Publish", "Publish selected pages?", "fa-check")
	def publish_pages(self, items):
		"""Publish selected pages"""
		service = DigitalStorefrontService(self.datamodel.session)
		
		count = 0
		for item in items:
			if service.publish_page(item.tenant_id, item.id, "system"):
				count += 1
		
		flash(f"Published {count} page(s)", "success")
		return redirect(self.get_redirect())

class StorefrontBannerModelView(ModelView):
	"""Storefront banner management view"""
	datamodel = SQLAInterface(PSStorefrontBanner)
	
	list_title = "Storefront Banners"
	show_title = "Banner Details"
	add_title = "Create Banner"
	edit_title = "Edit Banner"
	
	list_columns = ['title', 'location', 'storefront.name', 'is_active', 'sort_order', 'click_count', 'impression_count']
	show_columns = [
		'title', 'subtitle', 'description', 'storefront', 'location',
		'image_url', 'mobile_image_url', 'video_url',
		'cta_text', 'cta_url', 'cta_style',
		'start_date', 'end_date', 'is_active', 'sort_order',
		'click_count', 'impression_count',
		'background_color', 'text_color', 'overlay_opacity',
		'created_at', 'updated_at'
	]
	add_columns = [
		'storefront', 'title', 'subtitle', 'description', 'location',
		'image_url', 'mobile_image_url', 'video_url',
		'cta_text', 'cta_url', 'cta_style',
		'start_date', 'end_date', 'sort_order',
		'background_color', 'text_color', 'overlay_opacity'
	]
	edit_columns = [
		'title', 'subtitle', 'description', 'location',
		'image_url', 'mobile_image_url', 'video_url',
		'cta_text', 'cta_url', 'cta_style',
		'start_date', 'end_date', 'is_active', 'sort_order',
		'background_color', 'text_color', 'overlay_opacity'
	]
	
	search_columns = ['title', 'subtitle', 'description']
	
	formatters_columns = {
		'location': lambda x: f'<span class="badge badge-primary">{x.location.title()}</span>',
		'is_active': lambda x: '<i class="fa fa-check text-success"></i>' if x.is_active else '<i class="fa fa-times text-muted"></i>',
		'image_url': lambda x: f'<img src="{x.image_url}" height="50" />' if x.image_url else '-',
		'click_count': lambda x: f'<span class="badge badge-success">{x.click_count}</span>',
		'impression_count': lambda x: f'<span class="badge badge-info">{x.impression_count}</span>',
		'storefront.name': lambda x: x.storefront.name if x.storefront else '-'
	}

class StorefrontNavigationModelView(ModelView):
	"""Storefront navigation management view"""
	datamodel = SQLAInterface(PSStorefrontNavigation)
	
	list_title = "Navigation Menus"
	show_title = "Navigation Details"
	add_title = "Create Navigation Item"
	edit_title = "Edit Navigation Item"
	
	list_columns = ['label', 'location', 'storefront.name', 'parent.label', 'level', 'sort_order', 'is_active']
	show_columns = [
		'name', 'label', 'location', 'storefront', 'url', 'page',
		'parent', 'level', 'sort_order', 'is_active', 'is_featured',
		'open_in_new_tab', 'css_class', 'icon',
		'created_at', 'updated_at'
	]
	add_columns = [
		'storefront', 'name', 'label', 'location', 'url', 'page',
		'parent', 'sort_order', 'is_featured', 'open_in_new_tab',
		'css_class', 'icon'
	]
	edit_columns = [
		'name', 'label', 'location', 'url', 'page', 'parent',
		'sort_order', 'is_active', 'is_featured', 'open_in_new_tab',
		'css_class', 'icon'
	]
	
	search_columns = ['name', 'label', 'location']
	
	formatters_columns = {
		'location': lambda x: f'<span class="badge badge-primary">{x.location.title()}</span>',
		'level': lambda x: f'<span class="badge badge-secondary">L{x.level}</span>',
		'is_active': lambda x: '<i class="fa fa-check text-success"></i>' if x.is_active else '<i class="fa fa-times text-muted"></i>',
		'is_featured': lambda x: '<i class="fa fa-star text-warning"></i>' if x.is_featured else '-',
		'parent.label': lambda x: x.parent.label if x.parent else '-',
		'storefront.name': lambda x: x.storefront.name if x.storefront else '-'
	}

class StorefrontAnalyticsView(BaseView):
	"""Storefront analytics and reporting view"""
	
	route_base = '/storefrontanalytics'
	default_view = 'overview'
	
	@expose('/overview/')
	@has_access
	def overview(self):
		"""Storefront analytics overview"""
		# Get analytics data
		service = DigitalStorefrontService(self.appbuilder.get_session)
		
		# This would need to be enhanced with actual tenant context
		tenant_id = "default"  # In real implementation, get from session/context
		
		storefronts = service.list_storefronts(tenant_id)
		
		analytics_data = []
		for storefront in storefronts:
			data = service.get_storefront_analytics(tenant_id, storefront.id)
			analytics_data.append({
				'storefront': storefront,
				'analytics': data
			})
		
		self.update_redirect()
		return self.render_template(
			'platform_services/storefront_analytics.html',
			analytics_data=analytics_data,
			title="Storefront Analytics"
		)
	
	@expose('/performance/<storefront_id>/')
	@has_access
	def performance_details(self, storefront_id):
		"""Detailed performance metrics for a storefront"""
		service = DigitalStorefrontService(self.appbuilder.get_session)
		tenant_id = "default"  # In real implementation, get from session/context
		
		storefront = service.get_storefront(tenant_id, storefront_id)
		if not storefront:
			flash("Storefront not found", "error")
			return redirect(url_for('StorefrontAnalyticsView.overview'))
		
		analytics = service.get_storefront_analytics(tenant_id, storefront_id, days=30)
		
		# Get banner performance details
		banners = service.get_active_banners(tenant_id, storefront_id)
		
		self.update_redirect()
		return self.render_template(
			'platform_services/storefront_performance.html',
			storefront=storefront,
			analytics=analytics,
			banners=banners,
			title=f"Performance - {storefront.name}"
		)

# Widget for displaying storefront previews
class StorefrontPreviewWidget(ListThumbnail):
	"""Custom widget for displaying storefront previews"""
	
	template = 'platform_services/widgets/storefront_preview.html'
	
	def __call__(self, item, **kwargs):
		"""Render storefront preview"""
		return Markup(
			self.template_env.get_template(self.template).render(
				item=item, **kwargs
			)
		)