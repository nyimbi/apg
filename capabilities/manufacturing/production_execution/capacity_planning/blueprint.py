"""
Capacity Planning Blueprint

Flask blueprint for capacity planning functionality.
"""

from flask import Blueprint

capacity_planning_bp = Blueprint(
	'capacity_planning',
	__name__,
	url_prefix='/manufacturing/capacity-planning',
	template_folder='templates',
	static_folder='static'
)

@capacity_planning_bp.route('/')
def capacity_planning_home():
	"""Capacity planning home page"""
	return "Capacity Planning - Resource Optimization & Planning"

@capacity_planning_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "capacity_planning", "version": "1.0.0"}

__all__ = ["capacity_planning_bp"]