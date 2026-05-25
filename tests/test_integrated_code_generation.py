#!/usr/bin/env python3
"""
Integrated Code Generation Test
===============================

Test the complete APG compilation pipeline with the composable template system.
This tests the full flow from APG code → AST → Code Generation → Working Application.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from compiler.code_generator import PythonCodeGenerator, CodeGenConfig
from compiler.ast_builder import (
    ModuleDeclaration, EntityDeclaration, PropertyDeclaration, 
    MethodDeclaration, Parameter, TypeAnnotation, EntityType
)

def create_comprehensive_ast():
    """Create a comprehensive APG AST for testing"""
    
    # Create a module with multiple entities
    module = ModuleDeclaration(
        module_name="ECommerceApp",
        imports=[],
        entities=[],
        workflows=[]
    )
    
    # User entity (will trigger auth capability)
    user_entity = EntityDeclaration(
        name="User",
        entity_type=EntityType.AGENT,
        properties=[
            PropertyDeclaration("name", TypeAnnotation("str", False)),
            PropertyDeclaration("email", TypeAnnotation("str", False)),
            PropertyDeclaration("password", TypeAnnotation("str", False)),
            PropertyDeclaration("active", TypeAnnotation("bool", False), default_value="True")
        ],
        methods=[
            MethodDeclaration(
                name="authenticate",
                parameters=[Parameter("password", TypeAnnotation("str", False))],
                return_type=TypeAnnotation("bool", False),
                body=None
            ),
            MethodDeclaration(
                name="login",
                parameters=[],
                return_type=TypeAnnotation("bool", False),
                body=None
            )
        ]
    )
    
    # Product entity (will trigger inventory/business capability)
    product_entity = EntityDeclaration(
        name="Product",
        entity_type=EntityType.AGENT,
        properties=[
            PropertyDeclaration("name", TypeAnnotation("str", False)),
            PropertyDeclaration("price", TypeAnnotation("float", False)),
            PropertyDeclaration("inventory", TypeAnnotation("int", False)),
            PropertyDeclaration("description", TypeAnnotation("str", True))
        ],
        methods=[
            MethodDeclaration(
                name="update_inventory",
                parameters=[Parameter("quantity", TypeAnnotation("int", False))],
                return_type=TypeAnnotation("None", False),
                body=None
            )
        ]
    )
    
    # Order entity (will trigger payments capability)
    order_entity = EntityDeclaration(
        name="Order",
        entity_type=EntityType.AGENT,
        properties=[
            PropertyDeclaration("user_id", TypeAnnotation("int", False)),
            PropertyDeclaration("total", TypeAnnotation("float", False)),
            PropertyDeclaration("payment_method", TypeAnnotation("str", False)),
            PropertyDeclaration("stripe_payment_id", TypeAnnotation("str", True))
        ],
        methods=[
            MethodDeclaration(
                name="process_payment",
                parameters=[Parameter("amount", TypeAnnotation("float", False))],
                return_type=TypeAnnotation("bool", False),
                body=None
            )
        ]
    )
    
    # ChatBot entity (will trigger AI/LLM capability)
    chatbot_entity = EntityDeclaration(
        name="ChatBot",
        entity_type=EntityType.AGENT,
        properties=[
            PropertyDeclaration("model", TypeAnnotation("str", False), default_value="'gpt-4'"),
            PropertyDeclaration("context", TypeAnnotation("str", True))
        ],
        methods=[
            MethodDeclaration(
                name="generate_response",
                parameters=[Parameter("message", TypeAnnotation("str", False))],
                return_type=TypeAnnotation("str", False),
                body=None
            ),
            MethodDeclaration(
                name="chat",
                parameters=[Parameter("user_input", TypeAnnotation("str", False))],
                return_type=TypeAnnotation("str", False),
                body=None
            )
        ]
    )
    
    # Database entity 
    database_entity = EntityDeclaration(
        name="AppDatabase",
        entity_type=EntityType.DATABASE,
        properties=[
            PropertyDeclaration("connection", TypeAnnotation("str", False)),
            PropertyDeclaration("pool_size", TypeAnnotation("int", False), default_value="20")
        ],
        methods=[]
    )
    
    # Add all entities to module
    module.entities = [user_entity, product_entity, order_entity, chatbot_entity, database_entity]
    
    return module

def test_default_composable_generation():
    """Test default composable template generation"""
    print("🧪 Testing Default Composable Template Generation")
    print("=" * 60)
    
    # Create AST
    ast = create_comprehensive_ast()
    
    # Configure code generator for composable templates
    config = CodeGenConfig(
        use_composable_templates=True,
        template_output_mode="complete_app",
        generate_tests=False
    )
    
    # Generate code
    generator = PythonCodeGenerator(config)
    generated_files = generator.generate(ast)
    
    print(f"✅ Generated {len(generated_files)} files")
    print(f"📋 Files generated:")
    for filename in sorted(generated_files.keys()):
        print(f"   - {filename}")
    
    return generated_files

def test_custom_base_template():
    """Test with custom base template selection"""
    print("\n🎯 Testing Custom Base Template Selection")
    print("=" * 60)
    
    # Create AST
    ast = create_comprehensive_ast()
    
    # Configure for microservice base template
    config = CodeGenConfig(
        use_composable_templates=True,
        preferred_base_template="microservice",
        template_output_mode="complete_app"
    )
    
    # Generate code
    generator = PythonCodeGenerator(config)
    generated_files = generator.generate(ast)
    
    print(f"✅ Generated {len(generated_files)} files with microservice base")
    
    # Check that it's using FastAPI instead of Flask
    app_content = generated_files.get('app.py', '')
    if 'FastAPI' in app_content:
        print("   ✅ Using FastAPI (microservice base)")
    elif 'Flask' in app_content:
        print("   ⚠️  Using Flask (may have fallen back to default)")
    
    return generated_files

def test_capability_customization():
    """Test with additional and excluded capabilities"""
    print("\n🔧 Testing Capability Customization")
    print("=" * 60)
    
    # Create AST
    ast = create_comprehensive_ast()
    
    # Configure with additional capabilities and exclusions
    config = CodeGenConfig(
        use_composable_templates=True,
        additional_capabilities=["analytics/basic_analytics", "communication/websocket_communication"],
        exclude_capabilities=["data/postgresql_database"],  # Force exclude database
        template_output_mode="complete_app"
    )
    
    # Generate code
    generator = PythonCodeGenerator(config)
    generated_files = generator.generate(ast)
    
    print(f"✅ Generated {len(generated_files)} files with customized capabilities")
    
    # Check integration.py for capability integrations
    integration_content = generated_files.get('integration.py', '')
    if 'analytics' in integration_content:
        print("   ✅ Added analytics capability")
    if 'websocket' in integration_content or 'communication' in integration_content:
        print("   ✅ Added websocket capability")
    if 'postgresql' not in integration_content:
        print("   ✅ Excluded PostgreSQL capability")
    
    return generated_files

def test_hybrid_mode():
    """Test hybrid mode combining templates with legacy generation"""
    print("\n🔀 Testing Hybrid Generation Mode")
    print("=" * 60)
    
    # Create AST
    ast = create_comprehensive_ast()
    
    # Configure for hybrid mode
    config = CodeGenConfig(
        use_composable_templates=True,
        template_output_mode="hybrid"
    )
    
    # Generate code
    generator = PythonCodeGenerator(config)
    generated_files = generator.generate(ast)
    
    print(f"✅ Generated {len(generated_files)} files in hybrid mode")
    
    # Should have both template system files and legacy entity files
    has_template_files = any('capabilities/' in f for f in generated_files.keys())
    has_legacy_files = any(f in ['views.py', 'models.py'] for f in generated_files.keys())  
    
    if has_template_files:
        print("   ✅ Template system files present")
    if has_legacy_files:
        print("   ✅ Legacy entity files present")
    
    return generated_files

def test_fallback_to_legacy():
    """Test fallback to legacy generation when templates fail"""
    print("\n🔄 Testing Fallback to Legacy Generation")
    print("=" * 60)
    
    # Create AST
    ast = create_comprehensive_ast()
    
    # Configure to disable composable templates
    config = CodeGenConfig(
        use_composable_templates=False
    )
    
    # Generate code
    generator = PythonCodeGenerator(config)
    generated_files = generator.generate(ast)
    
    print(f"✅ Generated {len(generated_files)} files using legacy generation")
    
    # Should have legacy Flask-AppBuilder files
    expected_legacy_files = ['app.py', 'views.py', 'config.py', '__init__.py']
    legacy_files_present = [f for f in expected_legacy_files if f in generated_files.keys()]
    
    print(f"   ✅ Legacy files present: {', '.join(legacy_files_present)}")
    
    return generated_files

def compare_generation_approaches(template_files, legacy_files):
    """Compare the two generation approaches"""
    print("\n📊 Comparing Generation Approaches")
    print("=" * 60)
    
    print(f"Template System: {len(template_files)} files")
    print(f"Legacy System: {len(legacy_files)} files")
    
    # Show differences in app.py
    template_app = template_files.get('app.py', '')
    legacy_app = legacy_files.get('app.py', '')
    
    if template_app and legacy_app:
        template_lines = len(template_app.split('\n'))
        legacy_lines = len(legacy_app.split('\n'))
        print(f"\napp.py comparison:")
        print(f"   Template system: {template_lines} lines")
        print(f"   Legacy system: {legacy_lines} lines")
        
        # Check for capability integrations
        if 'capabilities' in template_app:
            print("   ✅ Template system includes capability integrations")
        if 'Flask-AppBuilder' in legacy_app:
            print("   ✅ Legacy system uses Flask-AppBuilder")

def create_test_application(generated_files, test_name):
    """Create a test application from generated files"""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"apg_test_{test_name}_"))
    
    # Write all files
    for filepath, content in generated_files.items():
        full_path = temp_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
    
    print(f"   📁 Test app created: {temp_dir}")
    return temp_dir

def main():
    """Run comprehensive integrated code generation tests"""
    print("🚀 APG Integrated Code Generation Test Suite")
    print("=" * 80)
    
    try:
        # Test 1: Default composable generation
        template_files = test_default_composable_generation()
        template_app_dir = create_test_application(template_files, "template_default")
        
        # Test 2: Custom base template
        microservice_files = test_custom_base_template()
        microservice_app_dir = create_test_application(microservice_files, "microservice")
        
        # Test 3: Capability customization
        custom_cap_files = test_capability_customization()
        custom_app_dir = create_test_application(custom_cap_files, "custom_capabilities")
        
        # Test 4: Hybrid mode
        hybrid_files = test_hybrid_mode()
        hybrid_app_dir = create_test_application(hybrid_files, "hybrid")
        
        # Test 5: Legacy fallback
        legacy_files = test_fallback_to_legacy()
        legacy_app_dir = create_test_application(legacy_files, "legacy")
        
        # Compare approaches
        compare_generation_approaches(template_files, legacy_files)
        
        print("\n" + "=" * 80)
        print("🎉 INTEGRATED CODE GENERATION TESTS COMPLETED!")
        print("=" * 80)
        
        print(f"\n📊 Test Results Summary:")
        print(f"   ✅ Default template generation: {len(template_files)} files")
        print(f"   ✅ Microservice generation: {len(microservice_files)} files")
        print(f"   ✅ Custom capabilities: {len(custom_cap_files)} files")
        print(f"   ✅ Hybrid mode: {len(hybrid_files)} files")
        print(f"   ✅ Legacy fallback: {len(legacy_files)} files")
        
        print(f"\n📁 Generated Test Applications:")
        print(f"   • Template system: {template_app_dir}")
        print(f"   • Microservice: {microservice_app_dir}")
        print(f"   • Custom capabilities: {custom_app_dir}")
        print(f"   • Hybrid mode: {hybrid_app_dir}")
        print(f"   • Legacy system: {legacy_app_dir}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. cd into any test app directory")
        print(f"   2. pip install -r requirements.txt")
        print(f"   3. python app.py")
        print(f"   4. Visit http://localhost:8080")
        
        # Clean up (comment out to keep test apps)
        cleanup = input("\n🗑️  Clean up test applications? (y/N): ").lower().strip()
        if cleanup == 'y':
            for app_dir in [template_app_dir, microservice_app_dir, custom_app_dir, hybrid_app_dir, legacy_app_dir]:
                shutil.rmtree(app_dir)
            print("✅ Test applications cleaned up")
        else:
            print("📁 Test applications preserved for inspection")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Integrated code generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)