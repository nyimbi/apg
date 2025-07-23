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
from .semantic_analyzer import SemanticAnalyzer
from .code_generator import CodeGenerator
from .compiler import APGCompiler

__all__ = [
    'APGParser',
    'ASTBuilder', 
    'SemanticAnalyzer',
    'CodeGenerator',
    'APGCompiler',
]