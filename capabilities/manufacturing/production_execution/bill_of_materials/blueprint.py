"""
Bill of Materials (BOM) Management Blueprint

Flask blueprint for BOM management functionality.
"""

from flask import Blueprint

bom_management_bp = Blueprint(
	'bom_management',
	__name__,
	url_prefix='/manufacturing/bom',
	template_folder='templates',
	static_folder='static'
)

@bom_management_bp.route('/')
def bom_management_home():
	"""BOM management home page"""
	return "Bill of Materials Management - Product Structure & Components"

@bom_management_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "bom_management", "version": "1.0.0"}

__all__ = ["bom_management_bp"]