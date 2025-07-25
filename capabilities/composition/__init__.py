"""
APG Capabilities Composition Engine

The composition engine enables APG programmers to compose custom ERP solutions by
selecting specific sub-capabilities. This system provides:

- Auto-discovery of all available capabilities and sub-capabilities
- Validation of dependencies between sub-capabilities
- Industry-specific templates for quick deployment
- Dynamic Flask application generation
- Database schema composition and migration management
- Runtime composition queries and updates
- Comprehensive error handling and validation

Core Components:
- Registry: Auto-discovery and metadata management
- Composer: Main composition engine for creating applications
- Validator: Dependency validation and conflict resolution
- Templates: Industry-specific pre-configured compositions
- BlueprintManager: Dynamic Flask blueprint management
- DatabaseManager: Schema composition and migration management

The system makes the entire hierarchical capability architecture work together seamlessly.
"""

from .registry import (
	CapabilityRegistry,
	SubCapabilityMetadata,
	CapabilityMetadata,
	DependencyInfo,
	get_registry
)

from .composer import (
	CompositionEngine,
	CompositionConfig,
	CompositionResult,
	CompositionContext,
	get_composer
)

from .validator import (
	DependencyValidator,
	ValidationResult,
	ValidationError,
	ConflictResolution,
	get_validator
)

from .templates import (
	TemplateManager,
	IndustryTemplate,
	TemplateConfiguration,
	get_template_manager
)

from .blueprint_manager import (
	BlueprintManager,
	BlueprintConfiguration,
	get_blueprint_manager
)

from .database_manager import (
	DatabaseManager,
	SchemaComposition,
	MigrationPlan,
	get_database_manager
)

# Export version and metadata
__version__ = "1.0.0"
__author__ = "APG Development Team"

# Main composition interface
def compose_application(
	tenant_id: str,
	capabilities: list[str],
	industry_template: str | None = None,
	custom_config: dict | None = None
) -> CompositionResult:
	"""
	Main entry point for composing an APG application.
	
	Args:
		tenant_id: Unique tenant identifier
		capabilities: List of capability codes to include
		industry_template: Optional industry template to apply
		custom_config: Custom configuration overrides
		
	Returns:
		CompositionResult with Flask app, database schema, and metadata
	"""
	composer = get_composer()
	return composer.compose(
		tenant_id=tenant_id,
		capabilities=capabilities,
		industry_template=industry_template,
		custom_config=custom_config
	)

def discover_capabilities() -> dict[str, CapabilityMetadata]:
	"""Discover all available capabilities and their metadata."""
	registry = get_registry()
	return registry.discover_all()

def validate_composition(capabilities: list[str]) -> ValidationResult:
	"""Validate a capability composition for dependencies and conflicts."""
	validator = get_validator()
	return validator.validate_composition(capabilities)

def get_industry_templates() -> list[IndustryTemplate]:
	"""Get all available industry-specific templates."""
	template_manager = get_template_manager()
	return template_manager.list_templates()

# Export all public interfaces
__all__ = [
	# Main composition interface
	"compose_application",
	"discover_capabilities", 
	"validate_composition",
	"get_industry_templates",
	
	# Registry components
	"CapabilityRegistry",
	"SubCapabilityMetadata",
	"CapabilityMetadata", 
	"DependencyInfo",
	"get_registry",
	
	# Composer components
	"CompositionEngine",
	"CompositionConfig",
	"CompositionResult",
	"CompositionContext",
	"get_composer",
	
	# Validator components
	"DependencyValidator",
	"ValidationResult",
	"ValidationError",
	"ConflictResolution",
	"get_validator",
	
	# Template components
	"TemplateManager",
	"IndustryTemplate",
	"TemplateConfiguration", 
	"get_template_manager",
	
	# Blueprint components
	"BlueprintManager",
	"BlueprintConfiguration",
	"get_blueprint_manager",
	
	# Database components
	"DatabaseManager",
	"SchemaComposition",
	"MigrationPlan",
	"get_database_manager",
]