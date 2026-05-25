#!/usr/bin/env python3
"""
Test Generated Output
======================

Test to see what files the composition engine generates.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine, APGASTAnalyzer

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

def main():
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
    
    # Generate files
    files = engine.generate_application_files(context)
    
    # Show the app.py file
    print("Generated app.py:")
    print("=" * 50)
    print(files.get('app.py', 'No app.py generated'))
    
    print("\n\nGenerated config.py:")
    print("=" * 50)
    print(files.get('config.py', 'No config.py generated'))

if __name__ == '__main__':
    main()