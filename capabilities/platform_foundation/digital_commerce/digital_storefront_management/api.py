"""
Digital Storefront Management API

REST API endpoints for storefront management operations.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_restful import Api, Resource
from marshmallow import Schema, fields, validate, ValidationError
from typing import Dict, Any, Optional
import logging

from .service import DigitalStorefrontService
from .models import StorefrontStatus, ThemeType, PageType, StorefrontCreate, StorefrontUpdate, PageCreate, BannerCreate

logger = logging.getLogger(__name__)

# Create blueprint
storefront_api_bp = Blueprint('storefront_api', __name__, url_prefix='/api/v1/storefronts')
api = Api(storefront_api_bp)

# Marshmallow Schemas for API validation
class StorefrontSchema(Schema):
	"""Schema for storefront data"""
	id = fields.Str(dump_only=True)
	name = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	code = fields.Str(required=True, validate=validate.Length(min=1, max=50))
	domain = fields.Str(allow_none=True, validate=validate.Length(max=255))
	subdomain = fields.Str(allow_none=True, validate=validate.Length(max=100))
	description = fields.Str(allow_none=True)
	status = fields.Str(validate=validate.OneOf([s.value for s in StorefrontStatus]))
	is_primary = fields.Bool(dump_only=True)
	theme_id = fields.Str(allow_none=True)
	default_language = fields.Str(validate=validate.Length(max=10))
	default_currency = fields.Str(validate=validate.Length(max=3))
	timezone = fields.Str(validate=validate.Length(max=50))
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)

class PageSchema(Schema):
	"""Schema for page data"""
	id = fields.Str(dump_only=True)
	title = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	slug = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	page_type = fields.Str(validate=validate.OneOf([t.value for t in PageType]))
	content = fields.Str(allow_none=True)
	is_published = fields.Bool(dump_only=True)
	created_at = fields.DateTime(dump_only=True)
	updated_at = fields.DateTime(dump_only=True)

class BannerSchema(Schema):
	"""Schema for banner data"""
	id = fields.Str(dump_only=True)
	title = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	location = fields.Str(required=True, validate=validate.Length(min=1, max=50))
	image_url = fields.Url(allow_none=True)
	cta_text = fields.Str(allow_none=True, validate=validate.Length(max=100))
	cta_url = fields.Url(allow_none=True)
	is_active = fields.Bool()
	click_count = fields.Int(dump_only=True)
	impression_count = fields.Int(dump_only=True)
	created_at = fields.DateTime(dump_only=True)

def get_tenant_id() -> str:
	"""Get tenant ID from request context"""
	# In a real implementation, this would extract tenant from JWT token or session
	return request.headers.get('X-Tenant-ID', 'default')

def get_user_id() -> str:
	"""Get user ID from request context"""
	# In a real implementation, this would extract user from JWT token or session
	return request.headers.get('X-User-ID', 'system')

class StorefrontListResource(Resource):
	"""Resource for listing and creating storefronts"""
	
	def get(self):
		"""List storefronts"""
		try:
			tenant_id = get_tenant_id()
			status_filter = request.args.get('status')
			
			# Get database session (this would need proper session management)
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			status = StorefrontStatus(status_filter) if status_filter else None
			storefronts = service.list_storefronts(tenant_id, status)
			
			schema = StorefrontSchema(many=True)
			return {
				'success': True,
				'data': schema.dump(storefronts),
				'count': len(storefronts)
			}
			
		except Exception as e:
			logger.error(f"Error listing storefronts: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500
	
	def post(self):
		"""Create new storefront"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			# Validate input
			schema = StorefrontSchema()
			try:
				data = schema.load(request.json)
			except ValidationError as e:
				return {'success': False, 'errors': e.messages}, 400
			
			# Get database session
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			# Create storefront
			storefront_data = StorefrontCreate(**data)
			storefront = service.create_storefront(tenant_id, storefront_data, user_id)
			
			return {
				'success': True,
				'data': schema.dump(storefront),
				'message': 'Storefront created successfully'
			}, 201
			
		except ValueError as e:
			return {'success': False, 'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error creating storefront: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500

class StorefrontResource(Resource):
	"""Resource for individual storefront operations"""
	
	def get(self, storefront_id):
		"""Get storefront by ID"""
		try:
			tenant_id = get_tenant_id()
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			storefront = service.get_storefront(tenant_id, storefront_id)
			if not storefront:
				return {'success': False, 'error': 'Storefront not found'}, 404
			
			schema = StorefrontSchema()
			return {
				'success': True,
				'data': schema.dump(storefront)
			}
			
		except Exception as e:
			logger.error(f"Error getting storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500
	
	def put(self, storefront_id):
		"""Update storefront"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			# Validate input
			schema = StorefrontSchema(partial=True)
			try:
				data = schema.load(request.json)
			except ValidationError as e:
				return {'success': False, 'errors': e.messages}, 400
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			# Update storefront
			update_data = StorefrontUpdate(**data)
			storefront = service.update_storefront(tenant_id, storefront_id, update_data, user_id)
			
			if not storefront:
				return {'success': False, 'error': 'Storefront not found'}, 404
			
			return {
				'success': True,
				'data': schema.dump(storefront),
				'message': 'Storefront updated successfully'
			}
			
		except Exception as e:
			logger.error(f"Error updating storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500

class StorefrontPagesResource(Resource):
	"""Resource for storefront pages"""
	
	def get(self, storefront_id):
		"""List pages for a storefront"""
		try:
			tenant_id = get_tenant_id()
			page_type_filter = request.args.get('page_type')
			published_only = request.args.get('published', 'false').lower() == 'true'
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			page_type = PageType(page_type_filter) if page_type_filter else None
			pages = service.list_pages(tenant_id, storefront_id, page_type, published_only)
			
			schema = PageSchema(many=True)
			return {
				'success': True,
				'data': schema.dump(pages),
				'count': len(pages)
			}
			
		except Exception as e:
			logger.error(f"Error listing pages for storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500
	
	def post(self, storefront_id):
		"""Create new page for storefront"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			# Validate input
			schema = PageSchema()
			try:
				data = schema.load(request.json)
			except ValidationError as e:
				return {'success': False, 'errors': e.messages}, 400
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			# Create page
			page_data = PageCreate(**data)
			page = service.create_page(tenant_id, storefront_id, page_data, user_id)
			
			return {
				'success': True,
				'data': schema.dump(page),
				'message': 'Page created successfully'
			}, 201
			
		except ValueError as e:
			return {'success': False, 'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error creating page for storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500

class StorefrontBannersResource(Resource):
	"""Resource for storefront banners"""
	
	def get(self, storefront_id):
		"""List banners for a storefront"""
		try:
			tenant_id = get_tenant_id()
			location_filter = request.args.get('location')
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			banners = service.get_active_banners(tenant_id, storefront_id, location_filter)
			
			schema = BannerSchema(many=True)
			return {
				'success': True,
				'data': schema.dump(banners),
				'count': len(banners)
			}
			
		except Exception as e:
			logger.error(f"Error listing banners for storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500
	
	def post(self, storefront_id):
		"""Create new banner for storefront"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			# Validate input
			schema = BannerSchema()
			try:
				data = schema.load(request.json)
			except ValidationError as e:
				return {'success': False, 'errors': e.messages}, 400
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			# Create banner
			banner_data = BannerCreate(**data)
			banner = service.create_banner(tenant_id, storefront_id, banner_data, user_id)
			
			return {
				'success': True,
				'data': schema.dump(banner),
				'message': 'Banner created successfully'
			}, 201
			
		except Exception as e:
			logger.error(f"Error creating banner for storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500

class StorefrontAnalyticsResource(Resource):
	"""Resource for storefront analytics"""
	
	def get(self, storefront_id):
		"""Get analytics for a storefront"""
		try:
			tenant_id = get_tenant_id()
			days = int(request.args.get('days', 30))
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			analytics = service.get_storefront_analytics(tenant_id, storefront_id, days)
			
			return {
				'success': True,
				'data': analytics
			}
			
		except Exception as e:
			logger.error(f"Error getting analytics for storefront {storefront_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500

class BannerTrackingResource(Resource):
	"""Resource for banner tracking"""
	
	def post(self, banner_id, action):
		"""Track banner interaction"""
		try:
			if action not in ['impression', 'click']:
				return {'success': False, 'error': 'Invalid action'}, 400
			
			from flask import g
			service = DigitalStorefrontService(g.db_session)
			
			if action == 'impression':
				result = service.track_banner_impression(banner_id)
			else:
				result = service.track_banner_click(banner_id)
			
			if result:
				return {'success': True, 'message': f'{action.title()} tracked successfully'}
			else:
				return {'success': False, 'error': 'Banner not found'}, 404
			
		except Exception as e:
			logger.error(f"Error tracking banner {action} for {banner_id}: {str(e)}")
			return {'success': False, 'error': 'Internal server error'}, 500

# Register API resources
api.add_resource(StorefrontListResource, '')
api.add_resource(StorefrontResource, '/<string:storefront_id>')
api.add_resource(StorefrontPagesResource, '/<string:storefront_id>/pages')
api.add_resource(StorefrontBannersResource, '/<string:storefront_id>/banners')
api.add_resource(StorefrontAnalyticsResource, '/<string:storefront_id>/analytics')
api.add_resource(BannerTrackingResource, '/banners/<string:banner_id>/track/<string:action>')

def init_api(app):
	"""Initialize the API with the Flask app"""
	app.register_blueprint(storefront_api_bp)