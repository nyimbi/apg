"""
Material Requirements Planning (MRP) Blueprint

Flask blueprint for MRP functionality including MRP runs,
material requirements calculation, and planned orders.
"""

from flask import Blueprint

mrp_bp = Blueprint(
	'mrp',
	__name__,
	url_prefix='/manufacturing/mrp',
	template_folder='templates',
	static_folder='static'
)

@mrp_bp.route('/')
def mrp_home():
	"""MRP home page"""
	return "Material Requirements Planning - MRP Engine & Planning"

@mrp_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "mrp", "version": "1.0.0"}

__all__ = ["mrp_bp"]