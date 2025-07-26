"""
Digital Storefront Management Blueprint

Flask blueprint registration for digital storefront management.
"""

from flask_appbuilder import AppBuilder

def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Digital Storefront Management sub-capability"""
	
	from .views import (
		StorefrontModelView, StorefrontThemeModelView, StorefrontPageModelView,
		StorefrontBannerModelView, StorefrontNavigationModelView, StorefrontAnalyticsView
	)
	from .models import (
		PSStorefront, PSStorefrontTheme, PSStorefrontPage,
		PSStorefrontBanner, PSStorefrontNavigation
	)
	
	# Register model views
	appbuilder.add_view(
		StorefrontModelView,
		"Storefronts",
		icon="fa-store",
		category="Platform Services",
		category_icon="fa-shopping-cart"
	)
	
	appbuilder.add_view(
		StorefrontThemeModelView,
		"Themes",
		icon="fa-palette",
		category="Platform Services"
	)
	
	appbuilder.add_view(
		StorefrontPageModelView,
		"Pages",
		icon="fa-file-alt",
		category="Platform Services"
	)
	
	appbuilder.add_view(
		StorefrontBannerModelView,
		"Banners",
		icon="fa-image",
		category="Platform Services"
	)
	
	appbuilder.add_view(
		StorefrontNavigationModelView,
		"Navigation",
		icon="fa-bars",
		category="Platform Services"
	)
	
	# Register analytics view
	appbuilder.add_view(
		StorefrontAnalyticsView,
		"Storefront Analytics",
		icon="fa-chart-line",
		category="Platform Services"
	)
	
	# Register API blueprint
	from .api import init_api
	init_api(appbuilder.app)