"""
Shop Floor Control Blueprint

Flask blueprint for shop floor control functionality.
"""

from flask import Blueprint

shop_floor_control_bp = Blueprint(
	'shop_floor_control',
	__name__,
	url_prefix='/manufacturing/shop-floor-control',
	template_folder='templates',
	static_folder='static'
)

@shop_floor_control_bp.route('/')
def shop_floor_control_home():
	"""Shop floor control home page"""
	return "Shop Floor Control - Real-time Production Monitoring"

@shop_floor_control_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "shop_floor_control", "version": "1.0.0"}

__all__ = ["shop_floor_control_bp"]