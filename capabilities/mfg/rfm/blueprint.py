"""
Recipe & Formula Management Blueprint

Flask blueprint for recipe and formula management functionality.
"""

from flask import Blueprint

recipe_formula_bp = Blueprint(
	'recipe_formula',
	__name__,
	url_prefix='/manufacturing/recipe-formula',
	template_folder='templates',
	static_folder='static'
)

@recipe_formula_bp.route('/')
def recipe_formula_home():
	"""Recipe & formula management home page"""
	return "Recipe & Formula Management - Process Manufacturing"

@recipe_formula_bp.route('/health')
def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "capability": "recipe_formula", "version": "1.0.0"}

__all__ = ["recipe_formula_bp"]