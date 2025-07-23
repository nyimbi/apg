"""
APG Language Server Implementation
==================================

Language Server Protocol implementation for APG language features including:
- Syntax highlighting and validation
- IntelliSense and code completion
- Error diagnostics
- Symbol navigation and references
- Hover information and documentation
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse
import json

# LSP Protocol implementation
try:
	from pygls.server import LanguageServer
	from pygls.protocol import LanguageServerProtocol
	from pygls.features import (
		COMPLETION, DEFINITION, HOVER, REFERENCES, DOCUMENT_SYMBOLS,
		DIAGNOSTIC, TEXT_DOCUMENT_DID_OPEN, TEXT_DOCUMENT_DID_CHANGE,
		TEXT_DOCUMENT_DID_SAVE, TEXT_DOCUMENT_DID_CLOSE, INITIALIZED
	)
	from lsprotocol.types import (
		CompletionItem, CompletionItemKind, CompletionParams,
		Location, Position, Range, Diagnostic, DiagnosticSeverity,
		DocumentSymbol, SymbolKind, DocumentSymbolParams,
		DefinitionParams, HoverParams, Hover, MarkupContent, MarkupKind,
		ReferenceParams, InitializeParams, InitializeResult,
		ServerCapabilities, TextDocumentSyncKind,
		CompletionOptions, HoverOptions, DefinitionOptions,
		ReferencesOptions, DocumentSymbolOptions
	)
	LSP_AVAILABLE = True
except ImportError:
	LSP_AVAILABLE = False
	# Create dummy classes for when LSP libraries aren't available
	class LanguageServer: pass
	class LanguageServerProtocol: pass

# APG Compiler imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from apg.compiler.parser import APGParser, APGSyntaxError
from apg.compiler.ast_builder import ASTBuilder, ModuleDeclaration, EntityDeclaration, PropertyDeclaration, MethodDeclaration
from apg.compiler.semantic_analyzer import SemanticAnalyzer, SemanticError, Symbol


# ========================================
# APG Language Server Implementation
# ========================================

class APGLanguageServer:
	"""
	APG Language Server providing IDE integration features.
	
	Features:
	- Real-time syntax validation
	- Intelligent code completion
	- Symbol definitions and references
	- Hover documentation
	- Document outline/symbols
	- Error diagnostics with suggestions
	"""
	
	def __init__(self):
		if not LSP_AVAILABLE:
			raise ImportError("Language Server libraries not available. Install with: pip install pygls lsprotocol")
		
		self.server = LanguageServer("apg-language-server", "1.0.0")
		self.parser = APGParser()
		self.ast_builder = ASTBuilder()
		self.semantic_analyzer = SemanticAnalyzer()
		
		# Document cache
		self.documents: Dict[str, str] = {}
		self.parsed_documents: Dict[str, ModuleDeclaration] = {}
		self.diagnostics_cache: Dict[str, List[Diagnostic]] = {}
		
		# APG language keywords and built-ins
		self.keywords = [
			"module", "version", "description", "author", "license",
			"agent", "digital_twin", "workflow", "db", "api", "form",
			"import", "export", "from", "as",
			"str", "int", "float", "bool", "list", "dict", "void", "any",
			"if", "else", "for", "while", "return", "break", "continue",
			"true", "false", "null", "async", "await",
			"table", "schema", "trigger", "procedure", "function", "view",
			"vector", "embedding", "halfvec", "sparsevec"
		]
		
		self.builtin_functions = [
			"len", "str", "int", "float", "bool", "now", "log", "print",
			"query", "execute", "generate_embedding", "vector_similarity"
		]
		
		self.entity_types = ["agent", "digital_twin", "workflow", "db", "api", "form"]
		
		self._setup_handlers()
		
		self.logger = logging.getLogger(__name__)
	
	def _setup_handlers(self):
		"""Setup LSP message handlers"""
		
		@self.server.feature(INITIALIZED)
		def initialized(ls, params):
			"""Handle initialization complete"""
			self.logger.info("APG Language Server initialized")
		
		@self.server.feature(TEXT_DOCUMENT_DID_OPEN)
		async def did_open(ls, params):
			"""Handle document open"""
			uri = params.text_document.uri
			text = params.text_document.text
			self.documents[uri] = text
			await self._validate_document(uri, text)
		
		@self.server.feature(TEXT_DOCUMENT_DID_CHANGE)
		async def did_change(ls, params):
			"""Handle document changes"""
			uri = params.text_document.uri
			# Get the full text from the change
			if params.content_changes:
				text = params.content_changes[0].text
				self.documents[uri] = text
				await self._validate_document(uri, text)
		
		@self.server.feature(TEXT_DOCUMENT_DID_SAVE)
		async def did_save(ls, params):
			"""Handle document save"""
			uri = params.text_document.uri
			if uri in self.documents:
				await self._validate_document(uri, self.documents[uri])
		
		@self.server.feature(TEXT_DOCUMENT_DID_CLOSE)
		def did_close(ls, params):
			"""Handle document close"""
			uri = params.text_document.uri
			self.documents.pop(uri, None)
			self.parsed_documents.pop(uri, None)
			self.diagnostics_cache.pop(uri, None)
		
		@self.server.feature(COMPLETION)
		async def completion(ls, params: CompletionParams) -> List[CompletionItem]:
			"""Provide code completion"""
			return await self._get_completions(params)
		
		@self.server.feature(HOVER)
		async def hover(ls, params: HoverParams) -> Optional[Hover]:
			"""Provide hover information"""
			return await self._get_hover_info(params)
		
		@self.server.feature(DEFINITION)
		async def definition(ls, params: DefinitionParams) -> Optional[List[Location]]:
			"""Go to definition"""
			return await self._get_definition(params)
		
		@self.server.feature(REFERENCES)
		async def references(ls, params: ReferenceParams) -> Optional[List[Location]]:
			"""Find references"""
			return await self._get_references(params)
		
		@self.server.feature(DOCUMENT_SYMBOLS)
		async def document_symbols(ls, params: DocumentSymbolParams) -> List[DocumentSymbol]:
			"""Get document symbols for outline"""
			return await self._get_document_symbols(params)
	
	async def _validate_document(self, uri: str, text: str):
		"""Validate document and send diagnostics"""
		try:
			# Parse the document
			file_path = self._uri_to_path(uri)
			parse_result = self.parser.parse_string(text, file_path)
			
			diagnostics = []
			
			# Add syntax errors
			for error in parse_result.get('errors', []):
				diagnostic = Diagnostic(
					range=Range(
						start=Position(line=max(0, error.line - 1), character=error.column),
						end=Position(line=max(0, error.line - 1), character=error.column + 10)
					),
					message=error.message,
					severity=DiagnosticSeverity.Error,
					source="apg-parser"
				)
				diagnostics.append(diagnostic)
			
			# If parsing succeeded, do semantic analysis
			if parse_result.get('success', False) and parse_result.get('parse_tree'):
				ast = self.ast_builder.build_ast(parse_result['parse_tree'], file_path)
				if ast:
					self.parsed_documents[uri] = ast
					semantic_result = self.semantic_analyzer.analyze(ast)
					
					# Add semantic errors
					for error in semantic_result.get('errors', []):
						diagnostic = Diagnostic(
							range=Range(
								start=Position(line=max(0, error.node.line - 1), character=error.node.column),
								end=Position(line=max(0, error.node.line - 1), character=error.node.column + 10)
							),
							message=error.message,
							severity=DiagnosticSeverity.Error,
							source="apg-semantic"
						)
						diagnostics.append(diagnostic)
					
					# Add semantic warnings
					for warning in semantic_result.get('warnings', []):
						diagnostic = Diagnostic(
							range=Range(
								start=Position(line=max(0, warning.node.line - 1), character=warning.node.column),
								end=Position(line=max(0, warning.node.line - 1), character=warning.node.column + 10)
							),
							message=warning.message,
							severity=DiagnosticSeverity.Warning,
							source="apg-semantic"
						)
						diagnostics.append(diagnostic)
			
			# Cache and send diagnostics
			self.diagnostics_cache[uri] = diagnostics
			self.server.publish_diagnostics(uri, diagnostics)
			
		except Exception as e:
			self.logger.error(f"Error validating document {uri}: {e}")
			# Send error diagnostic
			error_diagnostic = Diagnostic(
				range=Range(
					start=Position(line=0, character=0),
					end=Position(line=0, character=10)
				),
				message=f"Language server error: {str(e)}",
				severity=DiagnosticSeverity.Error,
				source="apg-server"
			)
			self.server.publish_diagnostics(uri, [error_diagnostic])
	
	async def _get_completions(self, params: CompletionParams) -> List[CompletionItem]:
		"""Get code completion suggestions"""
		uri = params.text_document.uri
		position = params.position
		
		completions = []
		
		try:
			# Get current document text
			text = self.documents.get(uri, "")
			lines = text.split('\n')
			
			if position.line < len(lines):
				current_line = lines[position.line]
				line_prefix = current_line[:position.character]
				
				# Context-aware completions
				if self._is_in_entity_context(line_prefix):
					completions.extend(self._get_entity_completions())
				elif self._is_in_property_context(line_prefix):
					completions.extend(self._get_property_completions())
				elif self._is_in_method_context(line_prefix):
					completions.extend(self._get_method_completions())
				elif self._is_in_database_context(line_prefix):
					completions.extend(self._get_database_completions())
				else:
					# General completions
					completions.extend(self._get_keyword_completions())
					completions.extend(self._get_builtin_completions())
					
					# Add symbols from current document
					if uri in self.parsed_documents:
						completions.extend(self._get_symbol_completions(self.parsed_documents[uri]))
		
		except Exception as e:
			self.logger.error(f"Error getting completions: {e}")
		
		return completions
	
	async def _get_hover_info(self, params: HoverParams) -> Optional[Hover]:
		"""Get hover information for symbol under cursor"""
		uri = params.text_document.uri
		position = params.position
		
		try:
			# Get word at position
			text = self.documents.get(uri, "")
			word = self._get_word_at_position(text, position)
			
			if not word:
				return None
			
			# Check if it's a keyword
			if word in self.keywords:
				return self._create_keyword_hover(word)
			
			# Check if it's a builtin function
			if word in self.builtin_functions:
				return self._create_builtin_hover(word)
			
			# Check if it's a symbol in the current document
			if uri in self.parsed_documents:
				symbol_info = self._find_symbol_info(self.parsed_documents[uri], word)
				if symbol_info:
					return self._create_symbol_hover(symbol_info)
		
		except Exception as e:
			self.logger.error(f"Error getting hover info: {e}")
		
		return None
	
	async def _get_definition(self, params: DefinitionParams) -> Optional[List[Location]]:
		"""Go to symbol definition"""
		uri = params.text_document.uri
		position = params.position
		
		try:
			text = self.documents.get(uri, "")
			word = self._get_word_at_position(text, position)
			
			if not word or uri not in self.parsed_documents:
				return None
			
			# Find symbol definition
			definition_location = self._find_symbol_definition(self.parsed_documents[uri], word)
			if definition_location:
				return [Location(uri=uri, range=definition_location)]
		
		except Exception as e:
			self.logger.error(f"Error getting definition: {e}")
		
		return None
	
	async def _get_references(self, params: ReferenceParams) -> Optional[List[Location]]:
		"""Find all references to symbol"""
		uri = params.text_document.uri
		position = params.position
		
		try:
			text = self.documents.get(uri, "")
			word = self._get_word_at_position(text, position)
			
			if not word:
				return None
			
			# Find all references in current document
			references = self._find_symbol_references(text, word)
			return [Location(uri=uri, range=ref) for ref in references]
		
		except Exception as e:
			self.logger.error(f"Error getting references: {e}")
		
		return None
	
	async def _get_document_symbols(self, params: DocumentSymbolParams) -> List[DocumentSymbol]:
		"""Get document symbols for outline view"""
		uri = params.text_document.uri
		
		try:
			if uri not in self.parsed_documents:
				return []
			
			ast = self.parsed_documents[uri]
			return self._extract_document_symbols(ast)
		
		except Exception as e:
			self.logger.error(f"Error getting document symbols: {e}")
			return []
	
	# ========================================
	# Helper methods
	# ========================================
	
	def _uri_to_path(self, uri: str) -> str:
		"""Convert URI to file path"""
		parsed = urlparse(uri)
		return parsed.path
	
	def _get_word_at_position(self, text: str, position: Position) -> Optional[str]:
		"""Get word at cursor position"""
		lines = text.split('\n')
		if position.line >= len(lines):
			return None
		
		line = lines[position.line]
		if position.character >= len(line):
			return None
		
		# Find word boundaries
		start = position.character
		end = position.character
		
		# Move start backward to word beginning
		while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
			start -= 1
		
		# Move end forward to word end
		while end < len(line) and (line[end].isalnum() or line[end] == '_'):
			end += 1
		
		return line[start:end] if start < end else None
	
	def _is_in_entity_context(self, line_prefix: str) -> bool:
		"""Check if cursor is in entity declaration context"""
		return any(entity_type in line_prefix for entity_type in self.entity_types)
	
	def _is_in_property_context(self, line_prefix: str) -> bool:
		"""Check if cursor is in property declaration context"""
		return ':' in line_prefix and '=' not in line_prefix
	
	def _is_in_method_context(self, line_prefix: str) -> bool:
		"""Check if cursor is in method declaration context"""
		return '(' in line_prefix and ')' in line_prefix and '->' in line_prefix
	
	def _is_in_database_context(self, line_prefix: str) -> bool:
		"""Check if cursor is in database/DBML context"""
		db_keywords = ['table', 'schema', 'trigger', 'procedure', 'function', 'view']
		return any(keyword in line_prefix for keyword in db_keywords)
	
	def _get_keyword_completions(self) -> List[CompletionItem]:
		"""Get keyword completions"""
		return [
			CompletionItem(
				label=keyword,
				kind=CompletionItemKind.Keyword,
				detail=f"APG keyword: {keyword}"
			)
			for keyword in self.keywords
		]
	
	def _get_builtin_completions(self) -> List[CompletionItem]:
		"""Get builtin function completions"""
		return [
			CompletionItem(
				label=func,
				kind=CompletionItemKind.Function,
				detail=f"APG builtin function: {func}",
				insert_text=f"{func}($0)"
			)
			for func in self.builtin_functions
		]
	
	def _get_entity_completions(self) -> List[CompletionItem]:
		"""Get entity-specific completions"""
		return [
			CompletionItem(
				label="name",
				kind=CompletionItemKind.Property,
				detail="Entity name property",
				insert_text="name: str = \"$1\";"
			),
			CompletionItem(
				label="description",
				kind=CompletionItemKind.Property,
				detail="Entity description property",
				insert_text="description: str = \"$1\";"
			),
			CompletionItem(
				label="process",
				kind=CompletionItemKind.Method,
				detail="Main processing method",
				insert_text="process: () -> $1 = {\n\t$2\n};"
			)
		]
	
	def _get_property_completions(self) -> List[CompletionItem]:
		"""Get property type completions"""
		return [
			CompletionItem(
				label="str",
				kind=CompletionItemKind.TypeParameter,
				detail="String type"
			),
			CompletionItem(
				label="int",
				kind=CompletionItemKind.TypeParameter,
				detail="Integer type"
			),
			CompletionItem(
				label="float",
				kind=CompletionItemKind.TypeParameter,
				detail="Float type"
			),
			CompletionItem(
				label="bool",
				kind=CompletionItemKind.TypeParameter,
				detail="Boolean type"
			),
			CompletionItem(
				label="list[str]",
				kind=CompletionItemKind.TypeParameter,
				detail="List of strings"
			),
			CompletionItem(
				label="dict[str, any]",
				kind=CompletionItemKind.TypeParameter,
				detail="Dictionary type"
			)
		]
	
	def _get_method_completions(self) -> List[CompletionItem]:
		"""Get method-specific completions"""
		return [
			CompletionItem(
				label="return",
				kind=CompletionItemKind.Keyword,
				detail="Return statement",
				insert_text="return $1;"
			),
			CompletionItem(
				label="if",
				kind=CompletionItemKind.Snippet,
				detail="If statement",
				insert_text="if ($1) {\n\t$2\n}"
			),
			CompletionItem(
				label="for",
				kind=CompletionItemKind.Snippet,
				detail="For loop",
				insert_text="for ($1 in $2) {\n\t$3\n}"
			)
		]
	
	def _get_database_completions(self) -> List[CompletionItem]:
		"""Get database/DBML completions"""
		return [
			CompletionItem(
				label="table",
				kind=CompletionItemKind.Class,
				detail="Database table",
				insert_text="table $1 {\n\t$2\n}"
			),
			CompletionItem(
				label="serial [pk]",
				kind=CompletionItemKind.Property,
				detail="Primary key column",
				insert_text="id serial [pk]"
			),
			CompletionItem(
				label="varchar(255)",
				kind=CompletionItemKind.TypeParameter,
				detail="Variable character column"
			),
			CompletionItem(
				label="vector(1536)",
				kind=CompletionItemKind.TypeParameter,
				detail="Vector embedding column"
			)
		]
	
	def _get_symbol_completions(self, ast: ModuleDeclaration) -> List[CompletionItem]:
		"""Get symbol completions from AST"""
		completions = []
		
		for entity in ast.entities:
			# Add entity name
			completions.append(CompletionItem(
				label=entity.name,
				kind=CompletionItemKind.Class,
				detail=f"APG {entity.entity_type.value}: {entity.name}"
			))
			
			# Add entity properties
			for prop in entity.properties:
				completions.append(CompletionItem(
					label=prop.name,
					kind=CompletionItemKind.Property,
					detail=f"Property: {prop.type_annotation.type_name}"
				))
			
			# Add entity methods
			for method in entity.methods:
				completions.append(CompletionItem(
					label=method.name,
					kind=CompletionItemKind.Method,
					detail=f"Method: {method.name}()",
					insert_text=f"{method.name}($0)"
				))
		
		return completions
	
	def _create_keyword_hover(self, keyword: str) -> Hover:
		"""Create hover info for keyword"""
		descriptions = {
			"module": "Declares an APG module with version and metadata",
			"agent": "Defines an autonomous agent with properties and methods",
			"digital_twin": "Creates a digital twin representation of a physical entity",
			"workflow": "Defines a workflow with steps and execution logic",
			"db": "Declares a database with schema and tables",
			"str": "String data type",
			"int": "Integer data type",
			"float": "Floating-point number data type",
			"bool": "Boolean data type (true/false)",
			"list": "List/array data type",
			"dict": "Dictionary/map data type"
		}
		
		description = descriptions.get(keyword, f"APG keyword: {keyword}")
		
		return Hover(
			contents=MarkupContent(
				kind=MarkupKind.Markdown,
				value=f"**{keyword}**\n\n{description}"
			)
		)
	
	def _create_builtin_hover(self, function: str) -> Hover:
		"""Create hover info for builtin function"""
		descriptions = {
			"len": "Returns the length of a list, string, or dictionary",
			"str": "Converts value to string",
			"int": "Converts value to integer",
			"float": "Converts value to float",
			"bool": "Converts value to boolean",
			"now": "Returns current timestamp",
			"print": "Prints value to console",
			"query": "Executes database query",
			"execute": "Executes database command"
		}
		
		description = descriptions.get(function, f"APG builtin function: {function}")
		
		return Hover(
			contents=MarkupContent(
				kind=MarkupKind.Markdown,
				value=f"**{function}()**\n\n{description}"
			)
		)
	
	def _create_symbol_hover(self, symbol_info: Dict[str, Any]) -> Hover:
		"""Create hover info for symbol"""
		symbol_type = symbol_info.get('type', 'symbol')
		name = symbol_info.get('name', 'unknown')
		details = symbol_info.get('details', '')
		
		return Hover(
			contents=MarkupContent(
				kind=MarkupKind.Markdown,
				value=f"**{name}** ({symbol_type})\n\n{details}"
			)
		)
	
	def _find_symbol_info(self, ast: ModuleDeclaration, symbol_name: str) -> Optional[Dict[str, Any]]:
		"""Find information about a symbol in the AST"""
		# Check entities
		for entity in ast.entities:
			if entity.name == symbol_name:
				return {
					'name': entity.name,
					'type': entity.entity_type.value,
					'details': f"APG {entity.entity_type.value} entity"
				}
			
			# Check properties
			for prop in entity.properties:
				if prop.name == symbol_name:
					return {
						'name': prop.name,
						'type': 'property',
						'details': f"Property of type {prop.type_annotation.type_name}"
					}
			
			# Check methods
			for method in entity.methods:
				if method.name == symbol_name:
					return_type = method.return_type.type_name if method.return_type else 'void'
					params = ', '.join([f"{p.name}: {p.type_annotation.type_name}" for p in method.parameters])
					return {
						'name': method.name,
						'type': 'method',
						'details': f"Method({params}) -> {return_type}"
					}
		
		return None
	
	def _find_symbol_definition(self, ast: ModuleDeclaration, symbol_name: str) -> Optional[Range]:
		"""Find the definition location of a symbol"""
		# This would need to track line/column information from the AST
		# For now, return None as a placeholder
		return None
	
	def _find_symbol_references(self, text: str, symbol_name: str) -> List[Range]:
		"""Find all references to a symbol in text"""
		references = []
		lines = text.split('\n')
		
		for line_num, line in enumerate(lines):
			col = 0
			while True:
				col = line.find(symbol_name, col)
				if col == -1:
					break
				
				# Check if it's a whole word
				if ((col == 0 or not line[col-1].isalnum()) and 
					(col + len(symbol_name) >= len(line) or not line[col + len(symbol_name)].isalnum())):
					references.append(Range(
						start=Position(line=line_num, character=col),
						end=Position(line=line_num, character=col + len(symbol_name))
					))
				
				col += 1
		
		return references
	
	def _extract_document_symbols(self, ast: ModuleDeclaration) -> List[DocumentSymbol]:
		"""Extract document symbols from AST for outline view"""
		symbols = []
		
		# Module symbol
		module_symbol = DocumentSymbol(
			name=ast.name,
			kind=SymbolKind.Module,
			range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10)),
			selection_range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10)),
			children=[]
		)
		
		# Add entities as children
		for entity in ast.entities:
			entity_symbol = DocumentSymbol(
				name=entity.name,
				kind=SymbolKind.Class,
				range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10)),
				selection_range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10)),
				children=[]
			)
			
			# Add properties
			for prop in entity.properties:
				prop_symbol = DocumentSymbol(
					name=prop.name,
					kind=SymbolKind.Property,
					range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10)),
					selection_range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10))
				)
				entity_symbol.children.append(prop_symbol)
			
			# Add methods
			for method in entity.methods:
				method_symbol = DocumentSymbol(
					name=method.name,
					kind=SymbolKind.Method,
					range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10)),
					selection_range=Range(start=Position(line=0, character=0), end=Position(line=0, character=10))
				)
				entity_symbol.children.append(method_symbol)
			
			module_symbol.children.append(entity_symbol)
		
		symbols.append(module_symbol)
		return symbols
	
	def run(self, host: str = "127.0.0.1", port: int = 2087):
		"""Start the language server"""
		if not LSP_AVAILABLE:
			print("Error: Language Server libraries not available.")
			print("Install with: pip install pygls lsprotocol")
			return
		
		print(f"Starting APG Language Server on {host}:{port}")
		self.server.start_tcp(host, port)


# ========================================
# Server Capabilities Configuration
# ========================================

def create_server_capabilities() -> ServerCapabilities:
	"""Create server capabilities for initialization"""
	return ServerCapabilities(
		text_document_sync=TextDocumentSyncKind.Full,
		completion_provider=CompletionOptions(
			trigger_characters=['.', ':', '(', '[', '{']
		),
		hover_provider=HoverOptions(),
		definition_provider=DefinitionOptions(),
		references_provider=ReferencesOptions(),
		document_symbol_provider=DocumentSymbolOptions()
	)


# ========================================
# CLI Entry Point
# ========================================

def main():
	"""Main entry point for APG language server"""
	import argparse
	
	parser = argparse.ArgumentParser(description="APG Language Server")
	parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
	parser.add_argument("--port", type=int, default=2087, help="Port to bind to")
	parser.add_argument("--log-level", default="INFO", help="Log level")
	parser.add_argument("--stdio", action="store_true", help="Use stdio transport")
	
	args = parser.parse_args()
	
	# Configure logging
	logging.basicConfig(
		level=getattr(logging, args.log_level.upper()),
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)
	
	if not LSP_AVAILABLE:
		print("Error: Language Server Protocol libraries not installed.")
		print("Install with: pip install pygls lsprotocol")
		sys.exit(1)
	
	try:
		server = APGLanguageServer()
		if args.stdio:
			server.server.start_io()
		else:
			server.run(args.host, args.port)
	except KeyboardInterrupt:
		print("\nShutting down APG Language Server")
	except Exception as e:
		print(f"Error starting language server: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()