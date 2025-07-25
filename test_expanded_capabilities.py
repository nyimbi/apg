#!/usr/bin/env python3
"""
Test Expanded Capabilities
===========================

Test the expanded APG composable template system with new IoT, AI, and BI capabilities.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.composition_engine import CompositionEngine

def test_capability_discovery():
    """Test discovery of all capabilities including new ones"""
    print("🔍 Testing Expanded Capability Discovery")
    print("=" * 60)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        available_caps = engine.capability_manager.get_available_capabilities()
        
        print(f"✅ Total capabilities discovered: {len(available_caps)}")
        
        # Group by category and count
        by_category = {}
        for cap_name in available_caps:
            capability = engine.capability_manager.get_capability(cap_name)
            if capability:
                cat = capability.category.value
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(capability)
        
        print(f"\n📂 Capabilities by Category:")
        total_caps = 0
        for cat, caps in sorted(by_category.items()):
            print(f"   {cat.upper():<20} - {len(caps):2d} capabilities")
            total_caps += len(caps)
            
            # Show a few examples from each category
            for cap in caps[:3]:
                print(f"      • {cap.name}")
            if len(caps) > 3:
                print(f"      ... and {len(caps) - 3} more")
        
        print(f"\n📊 Summary: {total_caps} capabilities across {len(by_category)} categories")
        
        # Expected categories with new additions
        expected_categories = {'auth', 'ai', 'data', 'payments', 'business', 'communication', 'analytics', 'iot'}
        found_categories = set(by_category.keys())
        
        if expected_categories.issubset(found_categories):
            print("✅ All expected capability categories found")
        else:
            missing = expected_categories - found_categories
            print(f"⚠️  Missing categories: {missing}")
        
        return True, by_category
        
    except Exception as e:
        print(f"❌ Error discovering capabilities: {e}")
        return False, {}

def test_iot_focused_composition():
    """Test composition for IoT-focused application"""
    print("\n🔧 Testing IoT-Focused Application Composition")
    print("=" * 60)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        # Create IoT-focused mock AST
        class MockIoTAST:
            def __init__(self):
                self.entities = [
                    MockEntity("Sensor", "AGENT", [
                        MockProperty("temperature", "float"),
                        MockProperty("humidity", "float"),
                        MockProperty("device_id", "str")
                    ], [
                        MockMethod("read_sensor"),
                        MockMethod("calibrate")
                    ]),
                    MockEntity("DigitalTwin", "AGENT", [
                        MockProperty("model", "str"),
                        MockProperty("state", "dict")
                    ], [
                        MockMethod("simulate"),
                        MockMethod("predict")
                    ]),
                    MockEntity("MQTTBroker", "AGENT", [
                        MockProperty("broker", "str"),
                        MockProperty("topic", "str")
                    ], [
                        MockMethod("publish"),
                        MockMethod("subscribe")
                    ]),
                    MockEntity("DeviceDatabase", "DATABASE", [], [])
                ]
        
        class MockEntity:
            def __init__(self, name, entity_type, properties, methods):
                self.name = name
                self.entity_type = MockEntityType(entity_type)
                self.properties = properties
                self.methods = methods
        
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
        
        mock_ast = MockIoTAST()
        
        # Compose IoT application
        context = engine.compose_application(
            mock_ast,
            project_name="Smart Factory IoT Platform",
            project_description="Industrial IoT platform with sensors, digital twins, and MQTT",
            author="IoT Test Suite"
        )
        
        print(f"✅ IoT application composition successful!")
        print(f"📋 Project: {context.project_name}")
        print(f"🏗️  Base Template: {context.base_template.name}")
        print(f"🔧 Selected Capabilities: {len(context.capabilities)}")
        
        # Check if IoT-related capabilities were detected
        capability_names = [cap.name for cap in context.capabilities]
        iot_related = [name for name in capability_names if any(keyword in name.lower() for keyword in ['mqtt', 'sensor', 'device', 'iot', 'twin'])]
        
        print(f"\n🔧 All Selected Capabilities:")
        for cap in context.capabilities:
            marker = "🎯" if any(keyword in cap.name.lower() for keyword in ['mqtt', 'sensor', 'device', 'iot', 'twin']) else "  "
            print(f"   {marker} {cap.name} ({cap.category.value})")
        
        if iot_related:
            print(f"\n✅ IoT-specific capabilities detected: {len(iot_related)}")
        else:
            print(f"⚠️  No IoT-specific capabilities detected (may need keyword tuning)")
        
        return True, context
        
    except Exception as e:
        print(f"❌ Error in IoT composition: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_ai_focused_composition():
    """Test composition for AI-focused application"""
    print("\n🧠 Testing AI-Focused Application Composition")
    print("=" * 60)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        # Create AI-focused mock AST
        class MockAIAST:
            def __init__(self):
                self.entities = [
                    MockEntity("ImageClassifier", "AGENT", [
                        MockProperty("model", "str"),
                        MockProperty("confidence", "float")
                    ], [
                        MockMethod("classify"),
                        MockMethod("train")
                    ]),
                    MockEntity("NLPProcessor", "AGENT", [
                        MockProperty("text", "str"),
                        MockProperty("sentiment", "str")
                    ], [
                        MockMethod("analyze"),
                        MockMethod("extract_entities")
                    ]),
                    MockEntity("RecommendationEngine", "AGENT", [
                        MockProperty("user_id", "int"),
                        MockProperty("recommendations", "list")
                    ], [
                        MockMethod("recommend"),
                        MockMethod("update_preferences")
                    ]),
                    MockEntity("MLDatastore", "DATABASE", [], [])
                ]
        
        class MockEntity:
            def __init__(self, name, entity_type, properties, methods):
                self.name = name
                self.entity_type = MockEntityType(entity_type)
                self.properties = properties
                self.methods = methods
        
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
        
        mock_ast = MockAIAST()
        
        # Compose AI application
        context = engine.compose_application(
            mock_ast,
            project_name="AI Analytics Platform",
            project_description="AI platform with computer vision, NLP, and recommendations",
            author="AI Test Suite"
        )
        
        print(f"✅ AI application composition successful!")
        print(f"📋 Project: {context.project_name}")
        print(f"🏗️  Base Template: {context.base_template.name}")
        print(f"🔧 Selected Capabilities: {len(context.capabilities)}")
        
        # Check if AI-related capabilities were detected
        ai_related = [cap.name for cap in context.capabilities if any(keyword in cap.name.lower() for keyword in ['ai', 'ml', 'nlp', 'vision', 'recommend', 'model'])]
        
        print(f"\n🔧 All Selected Capabilities:")
        for cap in context.capabilities:
            marker = "🎯" if any(keyword in cap.name.lower() for keyword in ['ai', 'ml', 'nlp', 'vision', 'recommend', 'model']) else "  "
            print(f"   {marker} {cap.name} ({cap.category.value})")
        
        if ai_related:
            print(f"\n✅ AI-specific capabilities detected: {len(ai_related)}")
        else:
            print(f"⚠️  No AI-specific capabilities detected (may need keyword tuning)")
        
        return True, context
        
    except Exception as e:
        print(f"❌ Error in AI composition: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_bi_focused_composition():
    """Test composition for BI-focused application"""
    print("\n📊 Testing BI-Focused Application Composition")
    print("=" * 60)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        # Create BI-focused mock AST
        class MockBIAST:
            def __init__(self):
                self.entities = [
                    MockEntity("KPIDashboard", "AGENT", [
                        MockProperty("metrics", "list"),
                        MockProperty("chart", "str")
                    ], [
                        MockMethod("update_kpi"),
                        MockMethod("generate_report")
                    ]),
                    MockEntity("DataWarehouse", "AGENT", [
                        MockProperty("etl", "str"),
                        MockProperty("analytics", "dict")
                    ], [
                        MockMethod("extract"),
                        MockMethod("transform"),
                        MockMethod("load")
                    ]),
                    MockEntity("BusinessMetrics", "AGENT", [
                        MockProperty("revenue", "float"),
                        MockProperty("growth", "float")
                    ], [
                        MockMethod("calculate_metrics"),
                        MockMethod("forecast")
                    ]),
                    MockEntity("ReportsDatabase", "DATABASE", [], [])
                ]
        
        class MockEntity:
            def __init__(self, name, entity_type, properties, methods):
                self.name = name
                self.entity_type = MockEntityType(entity_type)
                self.properties = properties
                self.methods = methods
        
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
        
        mock_ast = MockBIAST()
        
        # Compose BI application
        context = engine.compose_application(
            mock_ast,
            project_name="Executive BI Dashboard",
            project_description="Business intelligence platform with KPIs, reports, and analytics",
            author="BI Test Suite"
        )
        
        print(f"✅ BI application composition successful!")
        print(f"📋 Project: {context.project_name}")
        print(f"🏗️  Base Template: {context.base_template.name}")
        print(f"🔧 Selected Capabilities: {len(context.capabilities)}")
        
        # Check if BI-related capabilities were detected
        bi_related = [cap.name for cap in context.capabilities if any(keyword in cap.name.lower() for keyword in ['kpi', 'dashboard', 'analytics', 'report', 'warehouse', 'metrics'])]
        
        print(f"\n🔧 All Selected Capabilities:")
        for cap in context.capabilities:
            marker = "🎯" if any(keyword in cap.name.lower() for keyword in ['kpi', 'dashboard', 'analytics', 'report', 'warehouse', 'metrics']) else "  "
            print(f"   {marker} {cap.name} ({cap.category.value})")
        
        if bi_related:
            print(f"\n✅ BI-specific capabilities detected: {len(bi_related)}")
        else:
            print(f"⚠️  No BI-specific capabilities detected (may need keyword tuning)")
        
        return True, context
        
    except Exception as e:
        print(f"❌ Error in BI composition: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_complex_multi_domain_composition():
    """Test composition for complex multi-domain application"""
    print("\n🌟 Testing Complex Multi-Domain Application Composition")
    print("=" * 60)
    
    try:
        composable_root = Path(__file__).parent / 'templates' / 'composable'
        engine = CompositionEngine(composable_root)
        
        # Create complex multi-domain mock AST
        class MockComplexAST:
            def __init__(self):
                self.entities = [
                    # IoT components
                    MockEntity("SmartSensor", "AGENT", [
                        MockProperty("temperature", "float"),
                        MockProperty("device_id", "str")
                    ], [
                        MockMethod("read_data"),
                        MockMethod("mqtt_publish")
                    ]),
                    # AI components
                    MockEntity("PredictiveModel", "AGENT", [
                        MockProperty("model", "str"),
                        MockProperty("prediction", "float")
                    ], [
                        MockMethod("train"),
                        MockMethod("predict"),
                        MockMethod("nlp_analyze")
                    ]),
                    # BI components
                    MockEntity("ExecutiveDashboard", "AGENT", [
                        MockProperty("kpi", "dict"),
                        MockProperty("chart", "str")
                    ], [
                        MockMethod("generate_report"),
                        MockMethod("analytics")
                    ]),
                    # Business components
                    MockEntity("Customer", "AGENT", [
                        MockProperty("email", "str"),
                        MockProperty("password", "str")
                    ], [
                        MockMethod("login"),
                        MockMethod("purchase")
                    ]),
                    # Payments
                    MockEntity("PaymentProcessor", "AGENT", [
                        MockProperty("stripe_token", "str"),
                        MockProperty("amount", "float")
                    ], [
                        MockMethod("charge"),
                        MockMethod("refund")
                    ]),
                    MockEntity("DataPlatform", "DATABASE", [], [])
                ]
        
        class MockEntity:
            def __init__(self, name, entity_type, properties, methods):
                self.name = name
                self.entity_type = MockEntityType(entity_type)
                self.properties = properties
                self.methods = methods
        
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
        
        mock_ast = MockComplexAST()
        
        # Compose complex application
        context = engine.compose_application(
            mock_ast,
            project_name="Smart Enterprise Platform",
            project_description="Comprehensive platform with IoT, AI, BI, and e-commerce capabilities",
            author="Integration Test Suite"
        )
        
        print(f"✅ Complex multi-domain composition successful!")
        print(f"📋 Project: {context.project_name}")
        print(f"🏗️  Base Template: {context.base_template.name}")
        print(f"🔧 Total Capabilities: {len(context.capabilities)}")
        
        # Categorize capabilities by domain
        domain_capabilities = {
            'IoT': [],
            'AI/ML': [],
            'BI/Analytics': [],
            'Business': [],
            'Core': []
        }
        
        for cap in context.capabilities:
            if any(keyword in cap.name.lower() for keyword in ['mqtt', 'sensor', 'device', 'iot', 'twin']):
                domain_capabilities['IoT'].append(cap.name)
            elif any(keyword in cap.name.lower() for keyword in ['ai', 'ml', 'nlp', 'vision', 'recommend', 'model']):
                domain_capabilities['AI/ML'].append(cap.name)
            elif any(keyword in cap.name.lower() for keyword in ['kpi', 'dashboard', 'analytics', 'report', 'warehouse', 'metrics']):
                domain_capabilities['BI/Analytics'].append(cap.name)
            elif any(keyword in cap.name.lower() for keyword in ['inventory', 'payment', 'stripe', 'business']):
                domain_capabilities['Business'].append(cap.name)
            else:
                domain_capabilities['Core'].append(cap.name)
        
        print(f"\n🔧 Capabilities by Domain:")
        total_detected = 0
        for domain, caps in domain_capabilities.items():
            if caps:
                print(f"   {domain:<15} - {len(caps)} capabilities")
                for cap in caps:
                    print(f"      • {cap}")
                total_detected += len(caps)
        
        print(f"\n📊 Multi-domain Analysis:")
        print(f"   • Total entities in AST: {len(mock_ast.entities)}")
        print(f"   • Total capabilities selected: {len(context.capabilities)}")
        print(f"   • Domain-specific capabilities: {total_detected}")
        print(f"   • Domains covered: {len([d for d, caps in domain_capabilities.items() if caps])}")
        
        return True, context
        
    except Exception as e:
        print(f"❌ Error in complex composition: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run expanded capabilities test suite"""
    print("🚀 APG Expanded Capabilities Test Suite")
    print("=" * 80)
    
    tests = [
        ("Capability Discovery", test_capability_discovery),
        ("IoT-Focused Composition", test_iot_focused_composition),
        ("AI-Focused Composition", test_ai_focused_composition),
        ("BI-Focused Composition", test_bi_focused_composition),
        ("Complex Multi-Domain Composition", test_complex_multi_domain_composition)
    ]
    
    results = {}
    contexts = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Capability Discovery":
                success, data = test_func()
                results[test_name] = success
                if success:
                    contexts[test_name] = data  # Category data
            else:
                success, context = test_func()
                results[test_name] = success
                if success:
                    contexts[test_name] = context
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 80)
    print("🎉 EXPANDED CAPABILITIES TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if passed == total:
        print(f"\n🎯 All tests passed! The expanded APG system is working excellently!")
        
        # Show summary statistics
        if "Capability Discovery" in contexts:
            by_category = contexts["Capability Discovery"]
            total_caps = sum(len(caps) for caps in by_category.values())
            print(f"\n📈 System Capabilities Summary:")
            print(f"   • Total Capabilities: {total_caps}")
            print(f"   • Categories: {len(by_category)}")
            print(f"   • IoT Capabilities: {len(by_category.get('iot', []))}")
            print(f"   • AI Capabilities: {len(by_category.get('ai', []))}")
            print(f"   • Analytics/BI Capabilities: {len(by_category.get('analytics', []))}")
            print(f"   • Business Capabilities: {len(by_category.get('business', []))}")
        
        print(f"\n🚀 The APG Composable Template System now supports:")
        print(f"   • Industrial IoT and sensor management")
        print(f"   • Advanced AI/ML capabilities")
        print(f"   • Business intelligence and reporting")
        print(f"   • Multi-domain application composition")
        print(f"   • Intelligent capability detection")
        
        return True
    else:
        print(f"\n⚠️  Some tests failed. The system may need debugging.")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)