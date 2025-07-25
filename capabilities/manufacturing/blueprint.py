"""
Manufacturing Capability Blueprint Registration

Registers all manufacturing sub-capability blueprints and provides centralized routing
for the complete manufacturing ecosystem.
"""

from flask import Blueprint

# Import all sub-capability blueprints
from .production_planning.blueprint import production_planning_bp
from .material_requirements_planning.blueprint import mrp_bp
from .shop_floor_control.blueprint import shop_floor_control_bp
from .bill_of_materials.blueprint import bom_management_bp
from .capacity_planning.blueprint import capacity_planning_bp
from .quality_management.blueprint import quality_management_bp
from .recipe_formula_management.blueprint import recipe_formula_bp
from .manufacturing_execution_system.blueprint import mes_bp

# Main manufacturing blueprint
manufacturing_bp = Blueprint(
	'manufacturing',
	__name__,
	url_prefix='/manufacturing',
	template_folder='templates',
	static_folder='static'
)

@manufacturing_bp.route('/')
def manufacturing_dashboard():
	"""Manufacturing capability dashboard"""
	return "Manufacturing Dashboard - Production Control Center"

@manufacturing_bp.route('/health')
def health_check():
	"""Health check endpoint for manufacturing capability"""
	return {"status": "healthy", "capability": "manufacturing", "version": "1.0.0"}

# Register all sub-capability blueprints
def register_manufacturing_blueprints(app):
	"""Register all manufacturing sub-capability blueprints with the Flask app"""
	
	# Register main manufacturing blueprint
	app.register_blueprint(manufacturing_bp)
	
	# Register all sub-capability blueprints
	app.register_blueprint(production_planning_bp)
	app.register_blueprint(mrp_bp)
	app.register_blueprint(shop_floor_control_bp)
	app.register_blueprint(bom_management_bp)
	app.register_blueprint(capacity_planning_bp)
	app.register_blueprint(quality_management_bp)
	app.register_blueprint(recipe_formula_bp)
	app.register_blueprint(mes_bp)

__all__ = ["manufacturing_bp", "register_manufacturing_blueprints"]