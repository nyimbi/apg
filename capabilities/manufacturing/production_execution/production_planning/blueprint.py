"""
Production Planning Blueprint

Flask blueprint for production planning functionality including
master production schedules, production orders, and capacity planning.
"""

from flask import Blueprint

production_planning_bp = Blueprint(
	'production_planning',
	__name__,
	url_prefix='/manufacturing/production-planning',
	template_folder='templates',
	static_folder='static'
)

@production_planning_bp.route('/')
def production_planning_home():
	"""Production planning home page"""
	return "Production Planning - Master Scheduling & Order Management"

@production_planning_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "production_planning", "version": "1.0.0"}

__all__ = ["production_planning_bp"]