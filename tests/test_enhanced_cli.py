#!/usr/bin/env python3
"""
Enhanced CLI Test
=================

Test the enhanced APG CLI with composable template system commands.
"""

import sys
import subprocess
from pathlib import Path

def run_cli_command(args):
    """Run an APG CLI command and return the result"""
    try:
        result = subprocess.run(
            [sys.executable, 'cli.py'] + args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_cli_help():
    """Test CLI help command"""
    print("🧪 Testing CLI Help")
    print("=" * 50)
    
    success, stdout, stderr = run_cli_command(['--help'])
    
    if success:
        print("✅ CLI help command works")
        # Check if new commands are present
        if 'capabilities' in stdout and 'bases' in stdout and 'patterns' in stdout:
            print("✅ New composable template commands are available")
        else:
            print("⚠️  New commands may not be properly registered")
    else:
        print(f"❌ CLI help failed: {stderr}")
    
    return success

def test_capabilities_list():
    """Test capabilities list command"""
    print("\n🔧 Testing Capabilities List")
    print("=" * 50)
    
    success, stdout, stderr = run_cli_command(['capabilities', 'list'])
    
    if success:
        print("✅ Capabilities list command works")
        if 'AUTH' in stdout and 'DATA' in stdout:
            print("✅ Capabilities are being discovered")
        else:
            print("⚠️  Capabilities may not be loading properly")
    else:
        print(f"❌ Capabilities list failed: {stderr}")
    
    return success

def test_bases_list():
    """Test base templates list command"""
    print("\n🏗️  Testing Base Templates List")
    print("=" * 50)
    
    success, stdout, stderr = run_cli_command(['bases', 'list'])
    
    if success:
        print("✅ Base templates list command works")
        if 'flask_webapp' in stdout and 'microservice' in stdout:
            print("✅ Base templates are being discovered")
        else:
            print("⚠️  Base templates may not be loading properly")
    else:
        print(f"❌ Base templates list failed: {stderr}")
    
    return success

def test_patterns_list():
    """Test integration patterns list command"""
    print("\n🔗 Testing Integration Patterns List")
    print("=" * 50)
    
    success, stdout, stderr = run_cli_command(['patterns', 'list'])
    
    if success:
        print("✅ Integration patterns list command works")
        if 'ai_platform' in stdout and 'ecommerce_complete' in stdout:
            print("✅ Integration patterns are being discovered")
        else:
            print("⚠️  Integration patterns may not be loading properly")
    else:
        print(f"❌ Integration patterns list failed: {stderr}")
    
    return success

def test_capability_info():
    """Test capability info command"""
    print("\n📋 Testing Capability Info")
    print("=" * 50)
    
    success, stdout, stderr = run_cli_command(['capabilities', 'info', 'auth/basic_authentication'])
    
    if success:
        print("✅ Capability info command works")
        if 'Basic Authentication' in stdout and 'Flask-AppBuilder' in stdout:
            print("✅ Capability details are being shown")
        else:
            print("⚠️  Capability details may not be complete")
    else:
        print(f"❌ Capability info failed: {stderr}")
    
    return success

def test_cli_version():
    """Test CLI version"""
    print("\n📋 Testing CLI Version")
    print("=" * 50)
    
    success, stdout, stderr = run_cli_command(['--version'])
    
    if success:
        print("✅ CLI version command works")
        if 'APG CLI' in stdout:
            print("✅ Version information displayed")
    else:
        print(f"❌ CLI version failed: {stderr}")
    
    return success

def main():
    """Run enhanced CLI tests"""
    print("🚀 APG Enhanced CLI Test Suite")
    print("=" * 80)
    
    tests = [
        ("CLI Help", test_cli_help),
        ("CLI Version", test_cli_version),
        ("Capabilities List", test_capabilities_list),
        ("Base Templates List", test_bases_list),
        ("Integration Patterns List", test_patterns_list),
        ("Capability Info", test_capability_info),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 80)
    print("🎉 ENHANCED CLI TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if passed == total:
        print(f"\n🎯 All tests passed! The enhanced CLI is working correctly.")
        print(f"\n🚀 Available enhanced commands:")
        print(f"   • apg capabilities list")
        print(f"   • apg capabilities info <name>")
        print(f"   • apg capabilities add <name>")
        print(f"   • apg capabilities remove <name>")
        print(f"   • apg bases list")
        print(f"   • apg bases set <name>")
        print(f"   • apg patterns list")
        print(f"   • apg patterns apply <name>")
        
        return True
    else:
        print(f"\n⚠️  Some tests failed. The CLI may need debugging.")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)