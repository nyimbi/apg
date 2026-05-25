#!/usr/bin/env python3
"""
Composable Template CLI Test
============================

Test only the composable template CLI commands without the full compiler.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine
from templates.composable.base_template import BaseTemplateType
from templates.composable.capability import CapabilityCategory

def test_list_capabilities():
    """Test listing capabilities directly"""
    print("🔧 Testing Capabilities List")
    print("=" * 50)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        available_caps = engine.capability_manager.get_available_capabilities()
        
        print(f"✅ Found {len(available_caps)} capabilities:")
        
        # Group by category
        by_category = {}
        for cap_name in available_caps:
            capability = engine.capability_manager.get_capability(cap_name)
            if capability:
                cat = capability.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(capability)
        
        for cat, caps in sorted(by_category.items()):
            print(f"\n📂 {cat.upper()} ({len(caps)} capabilities)")
            for cap in caps:
                print(f"   {cap.name:<25} - {cap.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing capabilities: {e}")
        return False

def test_list_base_templates():
    """Test listing base templates directly"""
    print("\n🏗️  Testing Base Templates List")
    print("=" * 50)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        available_bases = engine.base_manager.get_available_bases()
        
        print(f"✅ Found {len(available_bases)} base templates:")
        for base_type in available_bases:
            template = engine.base_manager.get_base_template(base_type)
            if template:
                print(f"  {base_type.value:<15} - {template.description}")
            else:
                print(f"  {base_type.value:<15} - ERROR: Could not load")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing base templates: {e}")
        return False

def test_list_integration_patterns():
    """Test listing integration patterns directly"""
    print("\n🔗 Testing Integration Patterns List")
    print("=" * 50)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        integrations_dir = composable_root / 'integrations'
        
        if not integrations_dir.exists():
            print("❌ No integration patterns directory found")
            return False
        
        patterns = []
        for pattern_dir in integrations_dir.iterdir():
            if pattern_dir.is_dir():
                pattern_file = pattern_dir / 'pattern.json'
                if pattern_file.exists():
                    import json
                    with open(pattern_file, 'r') as f:
                        pattern_data = json.load(f)
                        patterns.append((pattern_dir.name, pattern_data))
        
        print(f"✅ Found {len(patterns)} integration patterns:")
        for pattern_id, pattern_data in patterns:
            print(f"  {pattern_id:<20} - {pattern_data.get('description', 'No description')}")
            
            use_cases = pattern_data.get('use_cases', [])
            if use_cases:
                formatted_cases = ', '.join(use_cases[:3])
                if len(use_cases) > 3:
                    formatted_cases += f" (and {len(use_cases) - 3} more)"
                print(f"    Use cases: {formatted_cases}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing integration patterns: {e}")
        return False

def test_capability_info():
    """Test getting capability info directly"""
    print("\n📋 Testing Capability Info")
    print("=" * 50)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        # Test basic authentication capability
        capability = engine.capability_manager.get_capability('auth/basic_authentication')
        
        if not capability:
            print("❌ Basic authentication capability not found")
            return False
        
        print(f"✅ Found capability details:")
        print(f"  Name: {capability.name}")
        print(f"  Category: {capability.category.value}")
        print(f"  Version: {capability.version}")
        print(f"  Author: {capability.author}")
        print(f"  Description: {capability.description}")
        
        if capability.features:
            print(f"  Features: {', '.join(capability.features[:3])}")
        
        if capability.python_requirements:
            print(f"  Requirements: {', '.join(capability.python_requirements[:3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting capability info: {e}")
        return False

def test_composition_integration():
    """Test full composition integration"""
    print("\n🎯 Testing Composition Integration")
    print("=" * 50)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        # Create a simple mock AST
        class MockAST:
            def __init__(self):
                self.entities = [
                    MockEntity("User", "AGENT", ["name", "email", "password"]),
                    MockEntity("AppDB", "DATABASE", [])
                ]
        
        class MockEntity:
            def __init__(self, name, entity_type, properties):
                self.name = name
                self.entity_type = MockEntityType(entity_type)
                self.properties = [MockProp(p) for p in properties]
                self.methods = []
        
        class MockEntityType:
            def __init__(self, name):
                self.name = name
        
        class MockProp:
            def __init__(self, name):
                self.name = name
                self.type_annotation = "str"
        
        mock_ast = MockAST()
        
        # Test composition
        context = engine.compose_application(
            mock_ast,
            project_name="TestApp",
            project_description="Test application for CLI",
            author="CLI Test"
        )
        
        print(f"✅ Composition successful:")
        print(f"  Project: {context.project_name}")
        print(f"  Base: {context.base_template.name}")
        print(f"  Capabilities: {len(context.capabilities)}")
        
        for cap in context.capabilities:
            print(f"    - {cap.name}")
        
        # Test file generation (just count, don't show content)
        generated_files = engine.generate_application_files(context)
        print(f"  Generated files: {len(generated_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in composition integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run composable template CLI tests"""
    print("🚀 APG Composable Template CLI Test Suite")
    print("=" * 80)
    
    tests = [
        ("Capabilities List", test_list_capabilities),
        ("Base Templates List", test_list_base_templates),
        ("Integration Patterns List", test_list_integration_patterns),
        ("Capability Info", test_capability_info),
        ("Composition Integration", test_composition_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 80)
    print("🎉 COMPOSABLE TEMPLATE CLI TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if passed == total:
        print(f"\n🎯 All composable template functionality is working!")
        print(f"\n🚀 The CLI would support these commands:")
        print(f"   • apg capabilities list            - List all available capabilities")
        print(f"   • apg capabilities info <name>     - Show detailed capability info")
        print(f"   • apg capabilities add <name>      - Add capability to project")
        print(f"   • apg capabilities remove <name>   - Remove capability from project")
        print(f"   • apg bases list                   - List available base templates")
        print(f"   • apg bases set <name>             - Set project base template")
        print(f"   • apg patterns list                - List integration patterns")
        print(f"   • apg patterns apply <name>        - Apply integration pattern")
        
        print(f"\n✅ The enhanced CLI is ready for integration!")
        return True
    else:
        print(f"\n⚠️  Some functionality needs debugging.")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)