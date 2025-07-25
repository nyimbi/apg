#!/usr/bin/env python3
"""
Test the APG Composition Engine
===============================

Simple test to verify the composition engine works correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine, APGASTAnalyzer
from templates.composable.base_template import BaseTemplateType

# Mock APG AST for testing
class MockAPGAST:
    def __init__(self):
        self.entities = [
            MockEntity("User", "AGENT", [
                MockProperty("name", "str"),
                MockProperty("email", "str"),
                MockProperty("password", "str")
            ]),
            MockEntity("UserDatabase", "DATABASE", [])
        ]

class MockEntity:
    def __init__(self, name, entity_type, properties=None, methods=None):
        self.name = name
        self.entity_type = MockEntityType(entity_type)
        self.properties = properties or []
        self.methods = methods or []

class MockEntityType:
    def __init__(self, name):
        self.name = name

class MockProperty:
    def __init__(self, name, type_annotation):
        self.name = name
        self.type_annotation = type_annotation

def test_composition_engine():
    """Test the composition engine with a simple AST"""
    print("🧪 Testing APG Composition Engine")
    print("=" * 50)
    
    # Initialize the composition engine
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    engine = CompositionEngine(composable_root)
    
    # Create mock AST
    mock_ast = MockAPGAST()
    
    # Test the composition
    context = engine.compose_application(
        mock_ast,
        project_name="Test App",
        project_description="A test application for the composition engine"
    )
    
    print(f"✅ Composition completed!")
    print(f"📋 Project: {context.project_name}")
    print(f"🏗️  Base Template: {context.base_template.name if context.base_template else 'None'}")
    print(f"🔧 Capabilities: {len(context.capabilities)}")
    
    for capability in context.capabilities:
        print(f"   - {capability.name} ({capability.category.value})")
    
    print(f"🤖 Agents: {len(context.apg_agents)}")
    print(f"🗄️  Databases: {len(context.apg_databases)}")
    
    # Test validation
    print("\n🔍 Validating composition...")
    validation = engine.validate_composition(context)
    print(f"❌ Errors: {len(validation['errors'])}")
    print(f"⚠️  Warnings: {len(validation['warnings'])}")
    print(f"ℹ️  Info: {len(validation['info'])}")
    
    for error in validation['errors']:
        print(f"   ERROR: {error}")
    for warning in validation['warnings']:
        print(f"   WARNING: {warning}")
    for info in validation['info']:
        print(f"   INFO: {info}")
    
    # Test file generation
    print("\n📄 Testing file generation...")
    try:
        files = engine.generate_application_files(context)
        print(f"✅ Generated {len(files)} files:")
        for filename in sorted(files.keys()):
            print(f"   - {filename}")
    except Exception as e:
        print(f"❌ File generation failed: {e}")
    
    print("\n🎉 Test completed successfully!")

if __name__ == '__main__':
    test_composition_engine()