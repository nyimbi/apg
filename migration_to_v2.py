"""
APG Platform Migration Script v1.0 -> v2.0

Comprehensive migration script to transition from the original APG capability
structure to the enhanced v2.0 architecture with minimal disruption.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

Usage:
    python migration_to_v2.py --action plan     # Plan migration without executing
    python migration_to_v2.py --action migrate  # Execute migration
    python migration_to_v2.py --action rollback # Rollback migration
"""

import os
import shutil
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler('migration_v2.log'),
		logging.StreamHandler()
	]
)
logger = logging.getLogger(__name__)

class APGMigrationV2:
	"""APG Platform migration manager for v2.0 architecture."""
	
	def __init__(self, capabilities_root: str = "."):
		self.capabilities_root = Path(capabilities_root).resolve()
		self.backup_dir = self.capabilities_root / "backup_v1"
		self.migration_log = []
		
		# Migration mapping from v1.0 to v2.0 structure
		self.migration_mapping = {
			# Core Business Operations (Reorganized)
			"core_financials": "core_business_operations/financial_management",
			"human_resources": "core_business_operations/human_capital_management",
			"procurement_purchasing": "core_business_operations/procurement_sourcing",
			"inventory_management": "core_business_operations/inventory_supply_chain",
			"supply_chain_management": "core_business_operations/inventory_supply_chain",  # Merged
			"sales_order_management": "core_business_operations/sales_revenue_management",
			
			# Manufacturing Production (Consolidated)
			"manufacturing": "manufacturing_production/production_execution",
			
			# Platform Foundation (Renamed)
			"platform_services": "platform_foundation/digital_commerce",
			
			# Industry Verticals (Enhanced)
			"pharmaceutical_specific": "industry_vertical_solutions/pharmaceutical_life_sciences",
			"mining_specific": "industry_vertical_solutions/mining_resources",
			
			# General Cross-Functional (Enhanced and Reorganized)
			"general_cross_functional/customer_relationship_management": 
				"general_cross_functional_enhanced/customer_relationship_management",
			"general_cross_functional/enterprise_asset_management": 
				"general_cross_functional_enhanced/enterprise_asset_management",
			"general_cross_functional/workflow_business_process_mgmt": 
				"general_cross_functional_enhanced/workflow_business_process_mgmt",
			"general_cross_functional/document_management": 
				"general_cross_functional_enhanced/document_content_management",
			"general_cross_functional/business_intelligence_analytics": 
				"general_cross_functional_enhanced/business_intelligence_analytics",
			"general_cross_functional/governance_risk_compliance": 
				"general_cross_functional_enhanced/governance_risk_compliance",
			"general_cross_functional/product_lifecycle_management": 
				"general_cross_functional_enhanced/product_lifecycle_management",
			
			# Emerging Technologies (Reorganized)
			"ai_orchestration": "emerging_technologies/artificial_intelligence",
			"general_cross_functional/computer_vision": "emerging_technologies/computer_vision_processing",
			"quantum_computing": "emerging_technologies/quantum_computing_research",
			"digital_twin": "emerging_technologies/digital_twin_simulation",
			"blockchain_security": "emerging_technologies/blockchain_distributed_ledger",
			
			# Composition (Enhanced)
			"composition": "composition_orchestration/capability_registry"
		}
		
		# New capabilities to create
		self.new_capabilities = {
			# New Cross-Functional
			"general_cross_functional_enhanced/geographical_location_services": {
				"template": "geographical_location_services",
				"description": "Comprehensive geofencing and location intelligence"
			},
			"general_cross_functional_enhanced/advanced_analytics_platform": {
				"template": "advanced_analytics_platform", 
				"description": "Self-service analytics and ML workbench"
			},
			"general_cross_functional_enhanced/integration_api_management": {
				"template": "integration_api_management",
				"description": "Enterprise integration hub and API gateway"
			},
			"general_cross_functional_enhanced/sustainability_esg_management": {
				"template": "sustainability_esg_management",
				"description": "ESG reporting and carbon footprint tracking"
			},
			"general_cross_functional_enhanced/knowledge_learning_management": {
				"template": "knowledge_learning_management",
				"description": "Corporate knowledge base and learning platform"
			},
			"general_cross_functional_enhanced/mobile_device_management": {
				"template": "mobile_device_management",
				"description": "Enterprise mobile apps and device policies"
			},
			"general_cross_functional_enhanced/multi_language_localization": {
				"template": "multi_language_localization",
				"description": "I18n/L10n and cultural adaptation"
			},
			
			# New Industry Verticals
			"industry_vertical_solutions/healthcare_medical": {
				"template": "healthcare_medical",
				"description": "HIPAA-compliant healthcare management"
			},
			"industry_vertical_solutions/energy_utilities": {
				"template": "energy_utilities",
				"description": "Smart grid and renewable energy management"
			},
			"industry_vertical_solutions/telecommunications": {
				"template": "telecommunications",
				"description": "Network operations and subscriber management"
			},
			"industry_vertical_solutions/transportation_logistics": {
				"template": "transportation_logistics",
				"description": "Fleet management and route optimization"
			},
			"industry_vertical_solutions/real_estate_facilities": {
				"template": "real_estate_facilities",
				"description": "Property management and facility optimization"
			},
			"industry_vertical_solutions/education_academic": {
				"template": "education_academic",
				"description": "Student information and academic management"
			},
			"industry_vertical_solutions/government_public_sector": {
				"template": "government_public_sector",
				"description": "Public administration and citizen services"
			},
			
			# New Emerging Technologies
			"emerging_technologies/machine_learning_data_science": {
				"template": "ml_data_science",
				"description": "Collaborative ML development environment"
			},
			"emerging_technologies/natural_language_processing": {
				"template": "nlp_processing",
				"description": "Text analytics and conversational AI"
			},
			"emerging_technologies/augmented_virtual_reality": {
				"template": "ar_vr_platform",
				"description": "Immersive experiences and spatial computing"
			},
			"emerging_technologies/robotic_process_automation": {
				"template": "rpa_automation",
				"description": "Intelligent automation and bot management"
			},
			"emerging_technologies/edge_computing_iot": {
				"template": "edge_iot_platform",
				"description": "Edge deployment and real-time processing"
			}
		}
	
	def create_backup(self) -> bool:
		"""Create complete backup of current structure."""
		try:
			logger.info("Creating backup of current capability structure...")
			
			if self.backup_dir.exists():
				shutil.rmtree(self.backup_dir)
			
			# Create backup directory
			self.backup_dir.mkdir(parents=True)
			
			# Copy current structure
			for item in self.capabilities_root.iterdir():
				if item.name not in ["backup_v1", "migration_to_v2.py", "__pycache__"]:
					if item.is_dir():
						shutil.copytree(item, self.backup_dir / item.name)
					else:
						shutil.copy2(item, self.backup_dir / item.name)
			
			# Create backup manifest
			manifest = {
				"backup_date": datetime.now().isoformat(),
				"original_structure": list(self.migration_mapping.keys()),
				"apg_version": "1.0.0",
				"migration_target": "2.0.0"
			}
			
			with open(self.backup_dir / "backup_manifest.json", "w") as f:
				json.dump(manifest, f, indent=2)
			
			logger.info(f"Backup created successfully at: {self.backup_dir}")
			return True
			
		except Exception as e:
			logger.error(f"Backup creation failed: {e}")
			return False
	
	def plan_migration(self) -> Dict[str, Any]:
		"""Plan migration without executing."""
		logger.info("Planning migration to v2.0 architecture...")
		
		plan = {
			"migration_actions": [],
			"new_capabilities": [],
			"deprecated_paths": [],
			"conflicts": [],
			"estimated_duration": "30-45 minutes"
		}
		
		# Plan moves and reorganizations
		for old_path, new_path in self.migration_mapping.items():
			old_full_path = self.capabilities_root / old_path
			new_full_path = self.capabilities_root / new_path
			
			if old_full_path.exists():
				plan["migration_actions"].append({
					"action": "move",
					"from": str(old_path),
					"to": str(new_path),
					"exists": True
				})
			else:
				plan["migration_actions"].append({
					"action": "missing",
					"from": str(old_path),
					"to": str(new_path),
					"exists": False
				})
		
		# Plan new capability creation
		for new_path, config in self.new_capabilities.items():
			plan["new_capabilities"].append({
				"path": new_path,
				"template": config["template"],
				"description": config["description"]
			})
		
		# Check for conflicts
		for old_path, new_path in self.migration_mapping.items():
			new_full_path = self.capabilities_root / new_path
			if new_full_path.exists() and (self.capabilities_root / old_path).exists():
				plan["conflicts"].append({
					"conflict": "destination_exists",
					"path": new_path,
					"resolution": "backup_and_merge"
				})
		
		logger.info(f"Migration plan created: {len(plan['migration_actions'])} moves, {len(plan['new_capabilities'])} new capabilities")
		return plan
	
	def execute_migration(self) -> bool:
		"""Execute the migration to v2.0."""
		try:
			logger.info("Starting migration to APG v2.0 architecture...")
			
			# Create backup first
			if not self.create_backup():
				logger.error("Migration aborted: Backup creation failed")
				return False
			
			# Create new directory structure
			self._create_new_directory_structure()
			
			# Move existing capabilities
			self._migrate_existing_capabilities()
			
			# Create new capabilities
			self._create_new_capabilities()
			
			# Update configuration files
			self._update_configuration_files()
			
			# Generate migration report
			self._generate_migration_report()
			
			logger.info("Migration to v2.0 completed successfully!")
			return True
			
		except Exception as e:
			logger.error(f"Migration failed: {e}")
			logger.info("Rolling back changes...")
			self.rollback_migration()
			return False
	
	def _create_new_directory_structure(self):
		"""Create the new v2.0 directory structure."""
		logger.info("Creating new directory structure...")
		
		new_directories = [
			"core_business_operations",
			"manufacturing_production", 
			"platform_foundation",
			"industry_vertical_solutions",
			"general_cross_functional_enhanced",
			"emerging_technologies",
			"composition_orchestration"
		]
		
		for directory in new_directories:
			dir_path = self.capabilities_root / directory
			dir_path.mkdir(parents=True, exist_ok=True)
			
			# Create __init__.py if it doesn't exist
			init_file = dir_path / "__init__.py"
			if not init_file.exists():
				with open(init_file, "w") as f:
					f.write(f'"""{directory.replace("_", " ").title()} capabilities."""\n')
	
	def _migrate_existing_capabilities(self):
		"""Migrate existing capabilities to new structure."""
		logger.info("Migrating existing capabilities...")
		
		for old_path, new_path in self.migration_mapping.items():
			old_full_path = self.capabilities_root / old_path
			new_full_path = self.capabilities_root / new_path
			
			if old_full_path.exists():
				logger.info(f"Moving {old_path} -> {new_path}")
				
				# Ensure parent directory exists
				new_full_path.parent.mkdir(parents=True, exist_ok=True)
				
				# Handle merging for inventory_supply_chain
				if "inventory_supply_chain" in new_path and new_full_path.exists():
					# Merge supply_chain_management into inventory_management
					self._merge_directories(old_full_path, new_full_path)
				else:
					# Regular move
					if new_full_path.exists():
						shutil.rmtree(new_full_path)
					shutil.move(str(old_full_path), str(new_full_path))
				
				self.migration_log.append(f"Moved: {old_path} -> {new_path}")
	
	def _merge_directories(self, source: Path, destination: Path):
		"""Merge two directories, combining their contents."""
		logger.info(f"Merging {source} into {destination}")
		
		for item in source.iterdir():
			dest_item = destination / item.name
			
			if item.is_dir():
				if dest_item.exists():
					# Recursively merge subdirectories
					self._merge_directories(item, dest_item)
				else:
					shutil.move(str(item), str(dest_item))
			else:
				# For files, if destination exists, rename source with suffix
				if dest_item.exists():
					dest_item = destination / f"{item.stem}_merged{item.suffix}"
				shutil.move(str(item), str(dest_item))
	
	def _create_new_capabilities(self):
		"""Create new capabilities from templates."""
		logger.info("Creating new capabilities...")
		
		for new_path, config in self.new_capabilities.items():
			capability_path = self.capabilities_root / new_path
			capability_path.mkdir(parents=True, exist_ok=True)
			
			# Create basic structure
			self._create_capability_template(capability_path, config)
			
			logger.info(f"Created new capability: {new_path}")
			self.migration_log.append(f"Created: {new_path}")
	
	def _create_capability_template(self, capability_path: Path, config: Dict[str, Any]):
		"""Create basic capability template structure."""
		template_name = config["template"]
		description = config["description"]
		
		# Create __init__.py
		init_content = f'''"""
{description}

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

__version__ = "1.0.0"
__description__ = "{description}"

# Capability metadata
CAPABILITY_METADATA = {{
    "capability_id": "{capability_path.name}",
    "version": __version__,
    "description": __description__,
    "status": "new_in_v2"
}}
'''
		
		with open(capability_path / "__init__.py", "w") as f:
			f.write(init_content)
		
		# Create models.py stub
		models_content = f'''"""
{description} - Data Models

© 2025 Datacraft. All rights reserved.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from uuid_extensions import uuid7str

# Model implementation placeholder
# TODO: Implement specific models for {template_name}

class {template_name.title().replace('_', '')}Model(BaseModel):
    """Base model for {description}."""
    id: str = Field(default_factory=uuid7str)
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
'''
		
		with open(capability_path / "models.py", "w") as f:
			f.write(models_content)
		
		# Create service.py stub
		service_content = f'''"""
{description} - Service Layer

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from typing import Any, Dict, List, Optional

class {template_name.title().replace('_', '')}Service:
    """Service layer for {description}."""
    
    def __init__(self):
        pass
    
    async def initialize(self) -> bool:
        """Initialize the service."""
        # TODO: Implement initialization logic
        return True
    
    async def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {{
            "service_name": "{template_name}",
            "description": "{description}",
            "status": "new_in_v2",
            "version": "1.0.0"
        }}
'''
		
		with open(capability_path / "service.py", "w") as f:
			f.write(service_content)
	
	def _update_configuration_files(self):
		"""Update configuration files for v2.0."""
		logger.info("Updating configuration files...")
		
		# Replace main __init__.py
		if (self.capabilities_root / "__init___NEW.py").exists():
			shutil.move(
				str(self.capabilities_root / "__init___NEW.py"),
				str(self.capabilities_root / "__init__.py")
			)
		
		# Replace main README
		if (self.capabilities_root / "README_NEW.md").exists():
			if (self.capabilities_root / "README.md").exists():
				shutil.move(
					str(self.capabilities_root / "README.md"),
					str(self.backup_dir / "README_v1.md")
				)
			shutil.move(
				str(self.capabilities_root / "README_NEW.md"),
				str(self.capabilities_root / "README.md")
			)
	
	def _generate_migration_report(self):
		"""Generate comprehensive migration report."""
		logger.info("Generating migration report...")
		
		report = {
			"migration_date": datetime.now().isoformat(),
			"source_version": "1.0.0",
			"target_version": "2.0.0",
			"actions_performed": self.migration_log,
			"new_capabilities_created": len(self.new_capabilities),
			"capabilities_migrated": len([action for action in self.migration_log if "Moved:" in action]),
			"total_capabilities": len(self.migration_mapping) + len(self.new_capabilities),
			"backup_location": str(self.backup_dir),
			"success": True
		}
		
		with open(self.capabilities_root / "migration_report_v2.json", "w") as f:
			json.dump(report, f, indent=2)
		
		logger.info(f"Migration report saved to: migration_report_v2.json")
	
	def rollback_migration(self) -> bool:
		"""Rollback migration to previous state."""
		try:
			logger.info("Rolling back migration...")
			
			if not self.backup_dir.exists():
				logger.error("No backup found for rollback")
				return False
			
			# Remove new structure
			for item in self.capabilities_root.iterdir():
				if item.name not in ["backup_v1", "migration_to_v2.py", "migration_report_v2.json"]:
					if item.is_dir():
						shutil.rmtree(item)
					else:
						item.unlink()
			
			# Restore from backup
			for item in self.backup_dir.iterdir():
				if item.name != "backup_manifest.json":
					if item.is_dir():
						shutil.copytree(item, self.capabilities_root / item.name)
					else:
						shutil.copy2(item, self.capabilities_root / item.name)
			
			logger.info("Rollback completed successfully")
			return True
			
		except Exception as e:
			logger.error(f"Rollback failed: {e}")
			return False
	
	def validate_migration(self) -> Dict[str, Any]:
		"""Validate migration results."""
		logger.info("Validating migration...")
		
		validation = {
			"structure_valid": True,
			"missing_capabilities": [],
			"unexpected_items": [],
			"init_files_present": True,
			"overall_status": "success"
		}
		
		# Check expected structure exists
		expected_dirs = [
			"core_business_operations",
			"manufacturing_production",
			"platform_foundation", 
			"industry_vertical_solutions",
			"general_cross_functional_enhanced",
			"emerging_technologies",
			"composition_orchestration"
		]
		
		for directory in expected_dirs:
			dir_path = self.capabilities_root / directory
			if not dir_path.exists():
				validation["structure_valid"] = False
				validation["missing_capabilities"].append(directory)
		
		# Validate key new capabilities
		key_new_capabilities = [
			"general_cross_functional_enhanced/geographical_location_services",
			"general_cross_functional_enhanced/advanced_analytics_platform",
			"industry_vertical_solutions/healthcare_medical"
		]
		
		for capability in key_new_capabilities:
			cap_path = self.capabilities_root / capability
			if not cap_path.exists():
				validation["missing_capabilities"].append(capability)
				validation["structure_valid"] = False
		
		if not validation["structure_valid"]:
			validation["overall_status"] = "failed"
		
		logger.info(f"Validation complete: {validation['overall_status']}")
		return validation

def main():
	"""Main migration script entry point."""
	parser = argparse.ArgumentParser(description="APG Platform Migration to v2.0")
	parser.add_argument(
		"--action",
		choices=["plan", "migrate", "rollback", "validate"],
		required=True,
		help="Migration action to perform"
	)
	parser.add_argument(
		"--capabilities-path",
		default=".",
		help="Path to capabilities directory"
	)
	
	args = parser.parse_args()
	
	migrator = APGMigrationV2(args.capabilities_path)
	
	if args.action == "plan":
		plan = migrator.plan_migration()
		print(json.dumps(plan, indent=2))
		
	elif args.action == "migrate":
		success = migrator.execute_migration()
		if success:
			print("✅ Migration to v2.0 completed successfully!")
		else:
			print("❌ Migration failed. Check logs for details.")
			
	elif args.action == "rollback":
		success = migrator.rollback_migration()
		if success:
			print("✅ Rollback completed successfully!")
		else:
			print("❌ Rollback failed. Check logs for details.")
			
	elif args.action == "validate":
		validation = migrator.validate_migration()
		print(json.dumps(validation, indent=2))

if __name__ == "__main__":
	main()