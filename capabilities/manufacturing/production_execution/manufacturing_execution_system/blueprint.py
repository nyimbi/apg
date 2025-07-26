"""
Manufacturing Execution System (MES) Blueprint

Flask blueprint for MES functionality.
"""

from flask import Blueprint

mes_bp = Blueprint(
	'mes',
	__name__,
	url_prefix='/manufacturing/mes',
	template_folder='templates',  
	static_folder='static'
)

@mes_bp.route('/')
def mes_home():
	"""MES home page"""
	return "Manufacturing Execution System - Real-time Operations Control"

@mes_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "mes", "version": "1.0.0"}

__all__ = ["mes_bp"]