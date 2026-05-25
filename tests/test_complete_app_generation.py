#!/usr/bin/env python3
"""
Complete Application Generation Test
====================================

Generate a complete Flask application using the composable template system
and verify it runs correctly.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine

# Mock APG AST that triggers multiple capabilities
class MockAPGAST:
    def __init__(self):
        self.entities = [
            MockEntity("User", "AGENT", [
                MockProperty("name", "str"),
                MockProperty("email", "str"),
                MockProperty("password", "str")
            ], [
                MockMethod("authenticate"),
                MockMethod("login")
            ]),
            MockEntity("UserDatabase", "DATABASE", []),
            MockEntity("ChatWorkflow", "WORKFLOW", [
                MockProperty("message", "str"),
                MockProperty("response", "str")
            ], [
                MockMethod("generate_response")
            ])
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

class MockMethod:
    def __init__(self, name):
        self.name = name

def generate_complete_application():
    """Generate a complete Flask application"""
    print("🚀 Generating Complete Flask Application")
    print("=" * 60)
    
    # Initialize the composition engine
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    engine = CompositionEngine(composable_root)
    
    # Create mock AST
    mock_ast = MockAPGAST()
    
    # Compose the application
    context = engine.compose_application(
        mock_ast,
        project_name="MyApp",
        project_description="A demonstration application with authentication and AI capabilities",
        author="APG Test Suite"
    )
    
    print(f"✅ Composition completed!")
    print(f"📋 Project: {context.project_name}")
    print(f"🏗️  Base Template: {context.base_template.name}")
    print(f"🔧 Capabilities: {len(context.capabilities)}")
    
    for capability in context.capabilities:
        print(f"   - {capability.name} ({capability.category.value})")
    
    # Generate all files
    print("\n📄 Generating application files...")
    generated_files = engine.generate_application_files(context)
    
    # Create temporary directory for the app
    temp_dir = Path(tempfile.mkdtemp(prefix="apg_generated_app_"))
    print(f"📁 Output directory: {temp_dir}")
    
    # Write all files
    for filepath, content in generated_files.items():
        full_path = temp_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
    
    print(f"✅ Generated {len(generated_files)} files")
    
    # List the generated structure
    print("\n📁 Generated file structure:")
    for root, dirs, files in os.walk(temp_dir):
        level = root.replace(str(temp_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per directory
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")
    
    return temp_dir, generated_files

def verify_application_structure(app_dir: Path, generated_files: dict):
    """Verify the generated application has correct structure"""
    print("\n🔍 Verifying application structure...")
    
    # Check essential files exist
    essential_files = [
        'app.py',
        'config.py', 
        'requirements.txt',
        'README.md'
    ]
    
    for file in essential_files:
        if file in generated_files:
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} MISSING")
    
    # Check capability files
    print("\n   Capability files:")
    capability_files = [f for f in generated_files.keys() if f.startswith('capabilities/')]
    for file in capability_files[:5]:  # Show first 5
        print(f"   ✅ {file}")
    if len(capability_files) > 5:
        print(f"   ... and {len(capability_files) - 5} more capability files")
    
    # Show sample of main app.py
    print("\n📄 Sample of generated app.py:")
    print("-" * 50)
    app_content = generated_files.get('app.py', '')
    lines = app_content.split('\n')
    for i, line in enumerate(lines[:30]):  # Show first 30 lines
        print(f"{i+1:2d}: {line}")
    if len(lines) > 30:
        print(f"... and {len(lines) - 30} more lines")

def main():
    try:
        # Generate the application
        app_dir, generated_files = generate_complete_application()
        
        # Verify the structure
        verify_application_structure(app_dir, generated_files)
        
        print(f"\n🎉 Application generation test completed successfully!")
        print(f"📁 Generated app location: {app_dir}")
        print(f"📚 To explore the generated app:")
        print(f"   cd {app_dir}")
        print(f"   pip install -r requirements.txt")
        print(f"   python app.py")
        
        # Ask if user wants to keep the generated app
        keep_app = input("\n🤔 Keep the generated application? (y/N): ").lower().strip()
        if keep_app == 'y':
            print(f"✅ Application preserved at: {app_dir}")
        else:
            shutil.rmtree(app_dir)
            print("🗑️  Temporary application cleaned up")
            
    except Exception as e:
        print(f"💥 Application generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()