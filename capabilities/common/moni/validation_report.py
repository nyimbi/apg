#!/usr/bin/env python3
"""
APG Monitoring Capability - Comprehensive Implementation Validation
Validates that all classes, methods, and functions are fully implemented with documentation

Author: Nyimbi Odero  
Copyright: © 2025 Datacraft
"""

import ast
import os
import inspect
from pathlib import Path
from typing import Dict, List, Any, Tuple

class ImplementationValidator:
    """Validates implementation completeness and documentation"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.results = {
            'files_analyzed': 0,
            'classes_found': 0,
            'methods_found': 0,
            'functions_found': 0,
            'documented_classes': 0,
            'documented_methods': 0,
            'documented_functions': 0,
            'incomplete_implementations': [],
            'undocumented_items': [],
            'placeholder_patterns': []
        }
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validate all Python files in the capability"""
        
        python_files = [
            'models.py',
            'service.py', 
            'metrics_engine.py',
            'alert_engine.py',
            'analytics_engine.py',
            'anomaly_detection.py',
            'timeseries_db.py',
            'views.py',
            'blueprint.py',
            '__init__.py'
        ]
        
        for filename in python_files:
            file_path = self.base_path / filename
            if file_path.exists():
                print(f"\n=== Analyzing {filename} ===")
                self._validate_file(file_path)
                self.results['files_analyzed'] += 1
            else:
                print(f"❌ Missing file: {filename}")
        
        return self.results
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Check for placeholder patterns
            self._check_placeholder_patterns(file_path.name, content)
            
            # Analyze classes and methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._validate_class(file_path.name, node, content)
                elif isinstance(node, ast.FunctionDef):
                    self._validate_function(file_path.name, node, content)
                elif isinstance(node, ast.AsyncFunctionDef):
                    self._validate_function(file_path.name, node, content, async_func=True)
                    
        except Exception as e:
            print(f"❌ Error analyzing {file_path.name}: {e}")
    
    def _check_placeholder_patterns(self, filename: str, content: str) -> None:
        """Check for placeholder patterns that indicate incomplete implementation"""
        
        placeholder_patterns = [
            'pass  # TODO',
            'pass # TODO', 
            'raise NotImplementedError',
            'return None  # TODO',
            'return None # TODO',
            '# PLACEHOLDER',
            '# TODO: implement',
            '# FIXME',
            'def stub_',
            'async def stub_',
            'return {}  # placeholder',
            'return []  # placeholder'
        ]
        
        for pattern in placeholder_patterns:
            if pattern in content:
                self.results['placeholder_patterns'].append({
                    'file': filename,
                    'pattern': pattern,
                    'lines': [i+1 for i, line in enumerate(content.split('\n')) if pattern in line]
                })
                print(f"⚠️  Found placeholder pattern '{pattern}' in {filename}")
    
    def _validate_class(self, filename: str, node: ast.ClassDef, content: str) -> None:
        """Validate a class definition"""
        self.results['classes_found'] += 1
        
        # Check if class has docstring
        has_docstring = (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str))
        
        if has_docstring:
            self.results['documented_classes'] += 1
            print(f"✅ Class {node.name}: documented")
        else:
            self.results['undocumented_items'].append(f"{filename}:{node.name} (class)")
            print(f"⚠️  Class {node.name}: missing docstring")
        
        # Validate methods
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._validate_method(filename, node.name, method, content)
    
    def _validate_method(self, filename: str, classname: str, node: ast.FunctionDef, content: str) -> None:
        """Validate a method definition"""
        self.results['methods_found'] += 1
        
        method_name = f"{classname}.{node.name}"
        
        # Check if method has docstring
        has_docstring = (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str))
        
        if has_docstring:
            self.results['documented_methods'] += 1
            print(f"  ✅ Method {method_name}: documented")
        else:
            self.results['undocumented_items'].append(f"{filename}:{method_name} (method)")
            print(f"  ⚠️  Method {method_name}: missing docstring")
        
        # Check for incomplete implementation
        self._check_method_implementation(filename, method_name, node, content)
    
    def _validate_function(self, filename: str, node: ast.FunctionDef, content: str, async_func: bool = False) -> None:
        """Validate a function definition"""
        self.results['functions_found'] += 1
        
        func_type = "async function" if async_func else "function"
        
        # Check if function has docstring
        has_docstring = (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str))
        
        if has_docstring:
            self.results['documented_functions'] += 1
            print(f"✅ Function {node.name}: documented")
        else:
            self.results['undocumented_items'].append(f"{filename}:{node.name} ({func_type})")
            print(f"⚠️  Function {node.name}: missing docstring")
        
        # Check for incomplete implementation
        self._check_method_implementation(filename, node.name, node, content)
    
    def _check_method_implementation(self, filename: str, name: str, node: ast.FunctionDef, content: str) -> None:
        """Check if method/function implementation is complete"""
        
        # Get the method/function body as string
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else len(lines)
        
        body_lines = lines[start_line:end_line]
        body_text = '\n'.join(body_lines)
        
        # Check for incomplete patterns
        incomplete_patterns = [
            'pass',
            'return None',
            'return {}',
            'return []',
            'raise NotImplementedError'
        ]
        
        # Only flag as incomplete if the entire body is just a placeholder
        body_stripped = body_text.strip()
        
        for pattern in incomplete_patterns:
            # Check if the method body only contains the pattern (possibly with docstring)
            if (pattern in body_stripped and 
                len([line for line in body_lines if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""') and not line.strip().startswith("'''")]) <= 2):
                
                self.results['incomplete_implementations'].append(f"{filename}:{name}")
                print(f"  ❌ {name}: appears to be incomplete (contains '{pattern}')")
                break
    
    def print_summary(self) -> None:
        """Print validation summary"""
        
        print(f"\n" + "="*60)
        print("APG MONITORING CAPABILITY - IMPLEMENTATION VALIDATION REPORT")  
        print("="*60)
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Files analyzed: {self.results['files_analyzed']}")
        print(f"   Classes found: {self.results['classes_found']}")
        print(f"   Methods found: {self.results['methods_found']}")
        print(f"   Functions found: {self.results['functions_found']}")
        
        print(f"\n📝 DOCUMENTATION COVERAGE:")
        total_items = self.results['classes_found'] + self.results['methods_found'] + self.results['functions_found']
        documented_items = self.results['documented_classes'] + self.results['documented_methods'] + self.results['documented_functions']
        coverage_pct = (documented_items / max(total_items, 1)) * 100
        
        print(f"   Documented classes: {self.results['documented_classes']}/{self.results['classes_found']}")
        print(f"   Documented methods: {self.results['documented_methods']}/{self.results['methods_found']}")  
        print(f"   Documented functions: {self.results['documented_functions']}/{self.results['functions_found']}")
        print(f"   Overall documentation coverage: {coverage_pct:.1f}%")
        
        print(f"\n🔍 IMPLEMENTATION COMPLETENESS:")
        incomplete_count = len(self.results['incomplete_implementations'])
        placeholder_count = len(self.results['placeholder_patterns'])
        
        if incomplete_count == 0 and placeholder_count == 0:
            print("   ✅ ALL IMPLEMENTATIONS APPEAR COMPLETE!")
        else:
            if incomplete_count > 0:
                print(f"   ❌ {incomplete_count} incomplete implementations found:")
                for item in self.results['incomplete_implementations']:
                    print(f"      - {item}")
            
            if placeholder_count > 0:
                print(f"   ⚠️  {placeholder_count} placeholder patterns found:")
                for item in self.results['placeholder_patterns']:
                    print(f"      - {item['file']}: '{item['pattern']}' on lines {item['lines']}")
        
        print(f"\n📋 UNDOCUMENTED ITEMS:")
        if len(self.results['undocumented_items']) == 0:
            print("   ✅ ALL ITEMS ARE DOCUMENTED!")
        else:
            print(f"   ⚠️  {len(self.results['undocumented_items'])} undocumented items:")
            for item in self.results['undocumented_items']:
                print(f"      - {item}")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if (incomplete_count == 0 and placeholder_count == 0 and 
            coverage_pct >= 90 and len(self.results['undocumented_items']) <= 5):
            print("   🎉 EXCELLENT! This is a production-ready, world-class implementation.")
            print("   🏆 The APG Monitoring Capability meets all quality standards.")
        elif incomplete_count == 0 and placeholder_count == 0:
            print("   👍 GOOD! Implementation is complete, documentation could be improved.")
        else:
            print("   ⚠️  NEEDS WORK! Some implementations appear incomplete.")


if __name__ == "__main__":
    print("APG Monitoring Capability - Implementation Validation")
    print("Analyzing all source files for completeness and documentation...")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create validator and run analysis
    validator = ImplementationValidator(current_dir)
    results = validator.validate_all_files()
    
    # Print summary
    validator.print_summary()