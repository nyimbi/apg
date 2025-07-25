#!/usr/bin/env python3
"""
Test Composition Engine with Authentication Capability
======================================================

Test the composition engine with the basic authentication capability.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine, APGASTAnalyzer

# Mock APG AST that should trigger authentication capability
class MockAPGAST:
    def __init__(self):
        self.entities = [
            MockEntity("User", "AGENT", [
                MockProperty("name", "str"),
                MockProperty("email", "str"),
                MockProperty("password", "str"),
                MockProperty("login", "method")
            ]),
            MockEntity("UserDatabase", "DATABASE", [])
        ]

class MockEntity:
    def __init__(self, name, entity_type, properties=None, methods=None):
        self.name = name
        self.entity_type = MockEntityType(entity_type)
        self.properties = properties or []
        self.methods = methods or [MockMethod("authenticate")]

class MockEntityType:
    def __init__(self, name):
        self.name = name

class MockProperty:
    def __init__(self, name, type_annotation):
        self.name = name
        self.type_annotation = type_annotation

class MockMethod:
    def __init__(self, name):
        self.name = name

def main():
    print("🧪 Testing APG Composition Engine with Authentication")
    print("=" * 60)
    
    # Initialize the composition engine
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    engine = CompositionEngine(composable_root)
    
    # Create mock AST with authentication keywords
    mock_ast = MockAPGAST()
    
    # Test the composition
    context = engine.compose_application(
        mock_ast,
        project_name="Auth Test App",
        project_description="A test application with authentication"
    )
    
    print(f"✅ Composition completed!")
    print(f"📋 Project: {context.project_name}")
    print(f"🏗️  Base Template: {context.base_template.name if context.base_template else 'None'}")
    print(f"🔧 Capabilities: {len(context.capabilities)}")
    
    for capability in context.capabilities:
        print(f"   - {capability.name} ({capability.category.value})")
    
    # Test file generation
    print("\n📄 Generating application files...")
    try:
        files = engine.generate_application_files(context) 
        print(f"✅ Generated {len(files)} files:")
        for filename in sorted(files.keys()):
            print(f"   - {filename}")
        
        # Show the app.py with capabilities
        print("\n" + "="*50)
        print("Generated app.py (with capabilities):")
        print("="*50)
        app_content = files.get('app.py', 'No app.py generated')
        # Show just the capability integration section
        lines = app_content.split('\n')
        in_capability_section = False
        for i, line in enumerate(lines):
            if 'Import capability modules' in line:
                in_capability_section = True
            if in_capability_section:
                print(line)
                if i < len(lines) - 1 and 'Application health check' in lines[i+1]:
                    break
        
    except Exception as e:
        print(f"❌ File generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()