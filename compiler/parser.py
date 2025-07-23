"""
APG Parser Module
================

Provides high-level parsing interface for APG source code using ANTLR-generated parsers.
Handles lexical analysis, syntax parsing, and initial AST construction.
"""

import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from antlr4.error.ErrorListener import ErrorListener

# Import generated ANTLR parsers (these will be available after running antlr)
sys.path.append(str(Path(__file__).parent.parent / "spec"))

try:
	from apgLexer import apgLexer
	from apgParser import apgParser
	from apgVisitor import apgVisitor
except ImportError as e:
	print(f"Warning: ANTLR-generated parsers not found: {e}")
	print("Please run: antlr -Dlanguage=Python3 -visitor apg.g4")
	apgLexer = apgParser = apgVisitor = None


class APGSyntaxError(Exception):
	"""Custom exception for APG syntax errors"""
	def __init__(self, message: str, line: int, column: int, source_file: Optional[str] = None):
		self.message = message
		self.line = line
		self.column = column
		self.source_file = source_file
		super().__init__(f"Syntax Error at {line}:{column}: {message}")


class APGErrorListener(ErrorListener):
	"""Custom error listener for collecting parsing errors"""
	
	def __init__(self):
		super().__init__()
		self.errors: List[APGSyntaxError] = []
	
	def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
		error = APGSyntaxError(msg, line, column)
		self.errors.append(error)


class APGParser:
	"""
	High-level APG parser that orchestrates lexical analysis and syntax parsing.
	
	Features:
	- Clean error handling and reporting
	- Source file tracking for debugging
	- Parse tree to AST conversion
	- Support for multiple input sources
	"""
	
	def __init__(self):
		self.error_listener = APGErrorListener()
		self._last_parse_tree = None
		self._last_tokens = None
	
	def parse_file(self, file_path: str) -> Dict[str, Any]:
		"""
		Parse an APG source file and return parse results.
		
		Args:
			file_path: Path to the APG source file
			
		Returns:
			Dictionary containing parse tree, errors, and metadata
		"""
		file_path = Path(file_path)
		if not file_path.exists():
			raise FileNotFoundError(f"APG source file not found: {file_path}")
		
		with open(file_path, 'r', encoding='utf-8') as f:
			source_code = f.read()
		
		return self.parse_string(source_code, str(file_path))
	
	def parse_string(self, source_code: str, source_name: str = "<string>") -> Dict[str, Any]:
		"""
		Parse APG source code from a string.
		
		Args:
			source_code: The APG source code to parse
			source_name: Name/path for error reporting
			
		Returns:
			Dictionary containing parse results and metadata
		"""
		if not apgLexer or not apgParser:
			raise RuntimeError("ANTLR parsers not available. Please generate them first.")
		
		# Reset error listener
		self.error_listener.errors.clear()
		
		# Create input stream and lexer
		input_stream = InputStream(source_code)
		lexer = apgLexer(input_stream)
		lexer.removeErrorListeners()
		lexer.addErrorListener(self.error_listener)
		
		# Create token stream and parser
		token_stream = CommonTokenStream(lexer)
		parser = apgParser(token_stream)
		parser.removeErrorListeners()
		parser.addErrorListener(self.error_listener)
		
		# Parse starting from the program rule
		try:
			parse_tree = parser.program()
			self._last_parse_tree = parse_tree
			self._last_tokens = token_stream
			
			return {
				'parse_tree': parse_tree,
				'tokens': token_stream,
				'errors': self.error_listener.errors.copy(),
				'source_name': source_name,
				'source_code': source_code,
				'success': len(self.error_listener.errors) == 0
			}
		
		except Exception as e:
			return {
				'parse_tree': None,
				'tokens': token_stream,
				'errors': [APGSyntaxError(f"Parser exception: {e}", 0, 0)],
				'source_name': source_name,
				'source_code': source_code,
				'success': False
			}
	
	def get_parse_errors(self) -> List[APGSyntaxError]:
		"""Get all parsing errors from the last parse operation"""
		return self.error_listener.errors.copy()
	
	def has_errors(self) -> bool:
		"""Check if the last parse operation had any errors"""
		return len(self.error_listener.errors) > 0
	
	def print_errors(self, file_path: Optional[str] = None):
		"""Print all parsing errors in a user-friendly format"""
		if not self.error_listener.errors:
			print("✓ No parsing errors")
			return
		
		print(f"✗ Found {len(self.error_listener.errors)} parsing error(s):")
		for error in self.error_listener.errors:
			location = f"{file_path or 'input'}:{error.line}:{error.column}"
			print(f"  {location}: {error.message}")


class APGParseTreeVisitor(apgVisitor if apgVisitor else object):
	"""
	Base visitor class for traversing APG parse trees.
	Extends the ANTLR-generated visitor with APG-specific functionality.
	"""
	
	def __init__(self):
		super().__init__()
		self.context_stack = []
		self.current_module = None
		self.current_entity = None
	
	def visit(self, tree):
		"""Override visit to add context tracking"""
		if hasattr(tree, 'getRuleIndex'):
			rule_name = self._get_rule_name(tree.getRuleIndex())
			self.context_stack.append(rule_name)
			try:
				result = super().visit(tree) if hasattr(super(), 'visit') else self.visitChildren(tree)
				return result
			finally:
				self.context_stack.pop()
		else:
			return super().visit(tree) if hasattr(super(), 'visit') else None
	
	def _get_rule_name(self, rule_index: int) -> str:
		"""Get the rule name from parser rule index"""
		if apgParser and hasattr(apgParser, 'ruleNames'):
			if 0 <= rule_index < len(apgParser.ruleNames):
				return apgParser.ruleNames[rule_index]
		return f"rule_{rule_index}"
	
	def get_current_context(self) -> List[str]:
		"""Get the current parsing context stack"""
		return self.context_stack.copy()
	
	def visitModule_declaration(self, ctx):
		"""Visit module declaration node"""
		if ctx.module_name():
			self.current_module = ctx.module_name().getText()
		return self.visitChildren(ctx)
	
	def visitEntity(self, ctx):
		"""Visit entity declaration node"""
		if ctx.IDENTIFIER():
			self.current_entity = ctx.IDENTIFIER().getText()
		result = self.visitChildren(ctx)
		self.current_entity = None
		return result


def test_parser():
	"""Simple test function for the APG parser"""
	parser = APGParser()
	
	# Test with simple APG code
	test_code = """
	module test version 1.0.0 {
		description: "Test module";
	}
	
	agent TestAgent {
		name: str = "Hello APG";
		process: () -> str = {
			return name;
		};
	}
	"""
	
	result = parser.parse_string(test_code, "test.apg")
	
	if result['success']:
		print("✓ Parser test successful!")
		print(f"Parse tree type: {type(result['parse_tree'])}")
	else:
		print("✗ Parser test failed:")
		for error in result['errors']:
			print(f"  {error}")


if __name__ == "__main__":
	test_parser()