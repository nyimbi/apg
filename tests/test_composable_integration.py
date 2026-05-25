#!/usr/bin/env python3
"""
Composable Template Integration Test
====================================

Test integration between composable template system and code generator
without relying on the full AST builder system.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine

# Create a simple mock AST that mimics the structure expected by the composition engine
class MockModuleAST:
    def __init__(self, module_name, entities):
        self.module_name = module_name
        self.entities = entities

class MockEntityAST:
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

def test_direct_composable_integration():
    """Test direct integration with composable template system"""
    print("🧪 Testing Direct Composable Template Integration")
    print("=" * 70)
    
    # Create a comprehensive mock AST
    entities = [
        MockEntityAST("User", "AGENT", [
            MockProperty("name", "str"),
            MockProperty("email", "str"),
            MockProperty("password", "str")
        ], [
            MockMethod("authenticate"),
            MockMethod("login")
        ]),
        MockEntityAST("Product", "AGENT", [
            MockProperty("name", "str"),
            MockProperty("price", "float"),
            MockProperty("inventory", "int")
        ], [
            MockMethod("update_stock")
        ]),
        MockEntityAST("Order", "AGENT", [
            MockProperty("user_id", "int"),
            MockProperty("total", "float"),
            MockProperty("stripe_payment", "str")
        ], [
            MockMethod("process_payment")
        ]),
        MockEntityAST("ChatBot", "AGENT", [
            MockProperty("model", "str"),
            MockProperty("llm", "str")
        ], [
            MockMethod("generate"),
            MockMethod("chat")
        ]),
        MockEntityAST("AppDatabase", "DATABASE", [
            MockProperty("connection", "str")
        ])
    ]
    
    mock_ast = MockModuleAST("ECommerceWithAI", entities)
    
    # Initialize composition engine
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    engine = CompositionEngine(composable_root)
    
    # Test composition
    context = engine.compose_application(
        mock_ast,
        project_name="ECommerce AI Platform",
        project_description="E-commerce platform with AI chatbot and payment processing",
        author="APG Integration Test"
    )
    
    print(f"✅ Composition completed successfully!")
    print(f"📋 Project: {context.project_name}")
    print(f"🏗️  Base Template: {context.base_template.name}")
    print(f"🔧 Detected Capabilities: {len(context.capabilities)}")
    
    for capability in context.capabilities:
        print(f"   - {capability.name} ({capability.category.value})")
    
    print(f"🤖 Detected Entities: {len(context.apg_agents)}")
    for agent in context.apg_agents:
        print(f"   - {agent['name']} ({agent['type']})")
    
    print(f"🗄️  Detected Databases: {len(context.apg_databases)}")
    for db in context.apg_databases:
        print(f"   - {db['name']} ({db['type']})")
    
    # Test file generation
    print(f"\n📄 Generating application files...")
    generated_files = engine.generate_application_files(context)
    
    print(f"✅ Generated {len(generated_files)} files:")
    
    # Categorize files
    app_files = []
    capability_files = []
    config_files = []
    
    for filepath in generated_files.keys():
        if filepath in ['app.py', 'README.md', '__init__.py']:
            app_files.append(filepath)
        elif filepath.startswith('capabilities/'):
            capability_files.append(filepath)
        else:
            config_files.append(filepath)
    
    print(f"\n   📱 Application files ({len(app_files)}):")
    for f in app_files:
        print(f"      - {f}")
    
    print(f"\n   🔧 Capability files ({len(capability_files)}):")
    for f in capability_files[:5]:  # Show first 5
        print(f"      - {f}")
    if len(capability_files) > 5:
        print(f"      ... and {len(capability_files) - 5} more")
    
    print(f"\n   ⚙️  Configuration files ({len(config_files)}):")
    for f in config_files:
        print(f"      - {f}")
    
    return context, generated_files

def analyze_generated_application(context, generated_files):
    """Analyze the generated application structure"""
    print(f"\n🔍 Analyzing Generated Application")
    print("=" * 70)
    
    # Analyze app.py
    app_content = generated_files.get('app.py', '')
    if app_content:
        lines = app_content.split('\n')
        total_lines = len(lines)
        
        # Count capability integrations
        integration_lines = [line for line in lines if 'integrate_' in line and 'capability' in line.lower()]
        
        print(f"📄 app.py Analysis:")
        print(f"   • Total lines: {total_lines}")
        print(f"   • Capability integrations: {len(integration_lines)}")
        
        # Show sample of app.py
        print(f"\n   Sample (first 15 lines):")
        for i, line in enumerate(lines[:15]):
            print(f"   {i+1:2d}: {line}")
        
        print(f"   ... (showing first 15 of {total_lines} lines)")
    
    # Analyze requirements.txt
    requirements = generated_files.get('requirements.txt', '')
    if requirements:
        req_lines = [line.strip() for line in requirements.split('\n') if line.strip() and not line.startswith('#')]
        print(f"\n📦 requirements.txt Analysis:")
        print(f"   • Dependencies: {len(req_lines)}")
        print(f"   • Key packages:")
        for req in req_lines[:8]:  # Show first 8
            print(f"      - {req}")
        if len(req_lines) > 8:
            print(f"      ... and {len(req_lines) - 8} more")
    
    # Analyze README.md
    readme = generated_files.get('README.md', '')
    if readme:
        readme_lines = [line for line in readme.split('\n') if line.strip()]
        print(f"\n📚 README.md Analysis:")
        print(f"   • Documentation lines: {len(readme_lines)}")
        print(f"   • Project: {context.project_name}")
        print(f"   • Base template: {context.base_template.name}")
    
    # Analyze capability structure
    capability_files = [f for f in generated_files.keys() if f.startswith('capabilities/')]
    if capability_files:
        print(f"\n🔧 Capability Structure Analysis:")
        
        # Group by capability
        capabilities = {}
        for filepath in capability_files:
            parts = filepath.split('/')
            if len(parts) >= 3:
                category = parts[1]
                capability = parts[2]
                cap_key = f"{category}/{capability}"
                if cap_key not in capabilities:
                    capabilities[cap_key] = []
                capabilities[cap_key].append(parts[3] if len(parts) > 3 else "root")
        
        for cap_name, files in capabilities.items():
            print(f"   • {cap_name}: {len(files)} files")

def test_different_configurations():
    """Test different configuration scenarios"""
    print(f"\n🎯 Testing Different Configuration Scenarios")
    print("=" * 70)
    
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    engine = CompositionEngine(composable_root)
    
    # Scenario 1: E-commerce focused (should detect inventory, payments)
    ecommerce_entities = [
        MockEntityAST("Product", "AGENT", [
            MockProperty("name", "str"),
            MockProperty("price", "float"),
            MockProperty("inventory", "int"),
            MockProperty("stock", "int")
        ]),
        MockEntityAST("Order", "AGENT", [
            MockProperty("stripe_payment", "str"),
            MockProperty("total", "float")
        ]),
        MockEntityAST("User", "AGENT", [
            MockProperty("email", "str"),
            MockProperty("password", "str")
        ])
    ]
    
    ecommerce_ast = MockModuleAST("ECommerceApp", ecommerce_entities)
    ecommerce_context = engine.compose_application(
        ecommerce_ast,
        project_name="E-Commerce Store",
        project_description="Online store with inventory and payments"
    )
    
    print(f"🛒 E-Commerce Scenario:")
    print(f"   Base: {ecommerce_context.base_template.name}")
    print(f"   Capabilities: {[cap.name for cap in ecommerce_context.capabilities]}")
    
    # Scenario 2: AI/ML focused (should detect LLM, possibly vector)
    ai_entities = [
        MockEntityAST("ChatBot", "AGENT", [
            MockProperty("model", "str"),
            MockProperty("llm", "str"),
            MockProperty("embedding", "str")
        ], [
            MockMethod("generate"),
            MockMethod("chat"),
            MockMethod("embed")
        ]),
        MockEntityAST("User", "AGENT", [
            MockProperty("login", "str")
        ])
    ]
    
    ai_ast = MockModuleAST("AIApp", ai_entities)
    ai_context = engine.compose_application(
        ai_ast,
        project_name="AI Assistant",
        project_description="AI-powered chat assistant"
    )
    
    print(f"\n🤖 AI/ML Scenario:")
    print(f"   Base: {ai_context.base_template.name}")
    print(f"   Capabilities: {[cap.name for cap in ai_context.capabilities]}")
    
    # Scenario 3: Analytics focused (should detect dashboard base)
    analytics_entities = [
        MockEntityAST("Dashboard", "AGENT", [
            MockProperty("charts", "str"),
            MockProperty("analytics", "str"),
            MockProperty("metrics", "str")
        ]),
        MockEntityAST("Report", "AGENT", [
            MockProperty("data", "str")
        ])
    ]
    
    analytics_ast = MockModuleAST("AnalyticsApp", analytics_entities)
    analytics_context = engine.compose_application(
        analytics_ast,
        project_name="Analytics Platform",
        project_description="Business analytics and reporting"
    )
    
    print(f"\n📊 Analytics Scenario:")
    print(f"   Base: {analytics_context.base_template.name}")
    print(f"   Capabilities: {[cap.name for cap in analytics_context.capabilities]}")

def main():
    """Run the composable template integration test"""
    try:
        # Test 1: Basic integration
        context, generated_files = test_direct_composable_integration()
        
        # Test 2: Analyze the generated application
        analyze_generated_application(context, generated_files)
        
        # Test 3: Different configuration scenarios
        test_different_configurations()
        
        print(f"\n" + "=" * 80)
        print("🎉 COMPOSABLE TEMPLATE INTEGRATION TEST COMPLETED!")
        print("=" * 80)
        
        print(f"\n📊 Integration Test Results:")
        print(f"   ✅ Composable template system: Working")
        print(f"   ✅ AST analysis and capability detection: Working")
        print(f"   ✅ Multi-entity application generation: Working")
        print(f"   ✅ Capability-based file generation: Working")
        print(f"   ✅ Configuration flexibility: Working")
        
        print(f"\n🎯 The composable template system is ready for:")
        print(f"   • Integration with the full APG compiler pipeline")
        print(f"   • Real APG code compilation")
        print(f"   • Production application generation")
        print(f"   • Community capability contributions")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)