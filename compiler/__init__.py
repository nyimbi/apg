"""
APG Language Compiler
====================

A comprehensive compiler and interpreter for the Application Programming Generation (APG) language.
This package provides parsing, semantic analysis, type checking, and code generation capabilities
for APG programs targeting multiple domains including IoT, AI/ML, business systems, and more.

Key Components:
- Parser: ANTLR-based parsing with full APG grammar support
- AST: Abstract Syntax Tree representation and manipulation
- Semantic: Type checking, symbol resolution, and semantic analysis
- CodeGen: Code generation for Python and other target languages
- Runtime: Runtime libraries and execution environment
"""

__version__ = "1.0.0"
__author__ = "APG Language Team"

from .parser import APGParser
from .ast_builder import ASTBuilder
from .baseline import build_compiler_baseline_report
from .capability_publish import build_capability_publish_report
from .deployment_verifier import build_deployment_verification_report
from .evidence_bundle import build_release_evidence_bundle
from .package_verifier import build_package_verification_report
from .semantic_analyzer import SemanticAnalyzer
from .diagnostics import audit_diagnostic_fixtures, diagnostic_registry, explain_diagnostic
from .drift import audit_drift_fixtures, build_drift_report, compare_semantic_models
from .explain import build_explain_report
from .graphs import audit_graph_fixtures
from .semantic_model import build_semantic_model, build_semantic_model_from_module
from .parser_golden import audit_parser_golden
from .packager import build_package_report
from .release import build_release_report
from .nl_plan import build_nl_plan
from .migrations import build_migration_plan
from .code_generator import CodeGenerator
from .compiler import APGCompiler

__all__ = [
    'APGParser',
    'ASTBuilder',
    'build_compiler_baseline_report',
    'build_capability_publish_report',
    'build_deployment_verification_report',
    'build_release_evidence_bundle',
    'build_package_verification_report',
    'SemanticAnalyzer',
    'audit_diagnostic_fixtures',
    'diagnostic_registry',
    'explain_diagnostic',
    'audit_drift_fixtures',
    'build_drift_report',
    'compare_semantic_models',
    'build_explain_report',
    'audit_graph_fixtures',
    'build_semantic_model',
    'build_semantic_model_from_module',
    'audit_parser_golden',
    'build_package_report',
    'build_release_report',
    'build_nl_plan',
    'build_migration_plan',
    'CodeGenerator',
    'APGCompiler',
]
