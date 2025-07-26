"""
Quality Management (QA/QC) Blueprint

Flask blueprint for quality management functionality.
"""

from flask import Blueprint

quality_management_bp = Blueprint(
	'quality_management',
	__name__,
	url_prefix='/manufacturing/quality-management',
	template_folder='templates',
	static_folder='static'
)

@quality_management_bp.route('/')
def quality_management_home():
	"""Quality management home page"""
	return "Quality Management - QA/QC & Compliance"

@quality_management_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "quality_management", "version": "1.0.0"}

__all__ = ["quality_management_bp"]