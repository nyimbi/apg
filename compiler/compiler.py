"""
APG Compiler Main Module
========================

High-level compiler interface that orchestrates parsing, semantic analysis, and code generation.
Provides a unified API for compiling APG source code to target languages and frameworks.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import logging

# Import compiler components
from .parser import APGParser, APGSyntaxError
from .ast_builder import ASTBuilder, ASTNode, ModuleDeclaration
from .semantic_analyzer import SemanticAnalyzer, SemanticError
from .code_generator import CodeGenerator, CodeGenConfig


# ========================================
# Compilation Result Types
# ========================================

class CompilationResult:
	"""Result of APG compilation process"""
	
	def __init__(self):
		self.success: bool = False
		self.module: Optional[ModuleDeclaration] = None
		self.generated_files: Dict[str, str] = {}
		self.errors: List[Union[APGSyntaxError, SemanticError]] = []
		self.warnings: List[Union[APGSyntaxError, SemanticError]] = []
		self.compilation_time: float = 0.0
		self.target_language: str = "python"
		self.output_directory: Optional[Path] = None
	
	def has_errors(self) -> bool:
		"""Check if compilation had any errors"""
		return len(self.errors) > 0
	
	def has_warnings(self) -> bool:
		"""Check if compilation had any warnings"""
		return len(self.warnings) > 0
	
	def print_summary(self):
		"""Print compilation summary"""
		if self.success:
			print(f"✓ Compilation successful in {self.compilation_time:.2f}s")
			print(f"  Generated {len(self.generated_files)} files for {self.target_language}")
			if self.output_directory:
				print(f"  Output: {self.output_directory}")
		else:
			print(f"✗ Compilation failed in {self.compilation_time:.2f}s")
		
		if self.errors:
			print(f"  {len(self.errors)} error(s)")
			for error in self.errors[:5]:  # Show first 5 errors
				print(f"    {error}")
			if len(self.errors) > 5:
				print(f"    ... and {len(self.errors) - 5} more errors")
		
		if self.warnings:
			print(f"  {len(self.warnings)} warning(s)")
			for warning in self.warnings[:3]:  # Show first 3 warnings
				print(f"    {warning}")
			if len(self.warnings) > 3:
				print(f"    ... and {len(self.warnings) - 3} more warnings")


# ========================================
# Main APG Compiler
# ========================================

class APGCompiler:
	"""
	Main APG compiler that orchestrates the full compilation pipeline.
	
	Compilation Pipeline:
	1. Lexical Analysis & Parsing (APGParser)
	2. AST Construction (ASTBuilder)
	3. Semantic Analysis (SemanticAnalyzer)  
	4. Code Generation (CodeGenerator)
	5. File Output
	
	Features:
	- Multi-target code generation (Python/Flask-AppBuilder, etc.)
	- Comprehensive error reporting and recovery
	- Incremental compilation support
	- Plugin architecture for extensions
	"""
	
	def __init__(self, config: CodeGenConfig = None):
		self.config = config or CodeGenConfig()
		self.logger = logging.getLogger(__name__)
		
		# Initialize compiler components
		self.parser = APGParser()
		self.ast_builder = ASTBuilder()
		self.semantic_analyzer = SemanticAnalyzer()
		self.code_generator = CodeGenerator(self.config)
		
		# Compilation state
		self.last_result: Optional[CompilationResult] = None
	
	def compile_file(self, source_file: Union[str, Path], 
					output_dir: Optional[Union[str, Path]] = None,
					target_language: str = None) -> CompilationResult:
		"""
		Compile a single APG source file.
		
		Args:
			source_file: Path to APG source file
			output_dir: Output directory for generated files
			target_language: Target language/framework
			
		Returns:
			CompilationResult with success status and generated files
		"""
		import time
		start_time = time.time()
		
		result = CompilationResult()
		result.target_language = target_language or self.config.target_language
		result.output_directory = Path(output_dir) if output_dir else None
		
		try:
			self.logger.info(f"Compiling APG file: {source_file}")
			
			# Phase 1: Parse source file
			parse_result = self._parse_file(source_file)
			if not parse_result['success']:
				result.errors.extend(parse_result['errors'])
				result.compilation_time = time.time() - start_time
				return result
			
			# Phase 2: Build AST
			ast = self._build_ast(parse_result, source_file)
			if not ast:
				result.errors.append(SemanticError("Failed to build AST", None, "ast"))
				result.compilation_time = time.time() - start_time
				return result
			
			result.module = ast
			
			# Phase 3: Semantic Analysis
			semantic_result = self._analyze_semantics(ast)
			result.errors.extend(semantic_result['errors'])
			result.warnings.extend(semantic_result['warnings'])
			
			# Continue code generation even with warnings, but not with errors
			if semantic_result['errors']:
				result.compilation_time = time.time() - start_time
				return result
			
			# Phase 4: Code Generation
			generated_files = self._generate_code(ast, result.target_language)
			result.generated_files = generated_files
			
			# Phase 5: Write output files if directory specified
			if result.output_directory:
				self._write_output_files(generated_files, result.output_directory)
			
			result.success = True
			result.compilation_time = time.time() - start_time
			self.last_result = result
			
			self.logger.info(f"Compilation successful: {len(generated_files)} files generated")
			return result
			
		except Exception as e:
			self.logger.error(f"Compilation failed with exception: {e}")
			result.errors.append(SemanticError(f"Internal compiler error: {e}", None, "internal"))
			result.compilation_time = time.time() - start_time
			return result
	
	def compile_string(self, source_code: str, 
					  module_name: str = "main",
					  target_language: str = None) -> CompilationResult:
		"""
		Compile APG source code from a string.
		
		Args:
			source_code: APG source code
			module_name: Name for the module
			target_language: Target language/framework
			
		Returns:
			CompilationResult with generated code
		"""
		import time
		start_time = time.time()
		
		result = CompilationResult()
		result.target_language = target_language or self.config.target_language
		
		try:
			self.logger.info(f"Compiling APG string: {module_name}")
			
			# Phase 1: Parse source string
			parse_result = self.parser.parse_string(source_code, module_name)
			if not parse_result['success']:
				result.errors.extend(parse_result['errors'])
				result.compilation_time = time.time() - start_time
				return result
			
			# Phase 2: Build AST
			ast = self.ast_builder.build_ast(parse_result['parse_tree'], module_name)
			if not ast:
				result.errors.append(SemanticError("Failed to build AST", None, "ast"))
				result.compilation_time = time.time() - start_time
				return result
			
			result.module = ast
			
			# Phase 3: Semantic Analysis
			semantic_result = self.semantic_analyzer.analyze(ast)
			result.errors.extend(semantic_result['errors'])
			result.warnings.extend(semantic_result['warnings'])
			
			if semantic_result['errors']:
				result.compilation_time = time.time() - start_time
				return result
			
			# Phase 4: Code Generation
			generated_files = self.code_generator.generate(ast, result.target_language)
			result.generated_files = generated_files
			
			result.success = True
			result.compilation_time = time.time() - start_time
			self.last_result = result
			
			self.logger.info(f"String compilation successful: {len(generated_files)} files generated")
			return result
			
		except Exception as e:
			self.logger.error(f"String compilation failed: {e}")
			result.errors.append(SemanticError(f"Internal compiler error: {e}", None, "internal"))
			result.compilation_time = time.time() - start_time
			return result
	
	def compile_project(self, project_dir: Union[str, Path],
					   output_dir: Optional[Union[str, Path]] = None,
					   target_language: str = None) -> List[CompilationResult]:
		"""
		Compile all APG files in a project directory.
		
		Args:
			project_dir: Directory containing APG source files
			output_dir: Output directory for generated files
			target_language: Target language/framework
			
		Returns:
			List of CompilationResults, one per source file
		"""
		project_path = Path(project_dir)
		if not project_path.exists():
			raise ValueError(f"Project directory does not exist: {project_path}")
		
		# Find all .apg files
		apg_files = list(project_path.glob("**/*.apg"))
		if not apg_files:
			self.logger.warning(f"No .apg files found in {project_path}")
			return []
		
		results = []
		self.logger.info(f"Compiling {len(apg_files)} APG files from {project_path}")
		
		for apg_file in apg_files:
			result = self.compile_file(apg_file, output_dir, target_language)
			results.append(result)
			
			if not result.success:
				self.logger.error(f"Failed to compile {apg_file}")
			else:
				self.logger.info(f"Successfully compiled {apg_file}")
		
		# Print summary
		successful = sum(1 for r in results if r.success)
		self.logger.info(f"Project compilation complete: {successful}/{len(results)} files successful")
		
		return results
	
	# ========================================
	# Internal Compilation Phases
	# ========================================
	
	def _parse_file(self, source_file: Union[str, Path]) -> Dict[str, Any]:
		"""Parse APG source file"""
		try:
			return self.parser.parse_file(str(source_file))
		except Exception as e:
			self.logger.error(f"Parsing failed for {source_file}: {e}")
			return {
				'success': False,
				'errors': [APGSyntaxError(f"Parse error: {e}", 0, 0, str(source_file))]
			}
	
	def _build_ast(self, parse_result: Dict[str, Any], source_file: Union[str, Path]) -> Optional[ModuleDeclaration]:
		"""Build AST from parse result"""
		try:
			return self.ast_builder.build_ast(
				parse_result['parse_tree'], 
				str(source_file)
			)
		except Exception as e:
			self.logger.error(f"AST building failed for {source_file}: {e}")
			return None
	
	def _analyze_semantics(self, ast: ModuleDeclaration) -> Dict[str, Any]:
		"""Perform semantic analysis on AST"""
		try:
			return self.semantic_analyzer.analyze(ast)
		except Exception as e:
			self.logger.error(f"Semantic analysis failed: {e}")
			return {
				'success': False,
				'errors': [SemanticError(f"Semantic analysis error: {e}", ast, "semantic")],
				'warnings': []
			}
	
	def _generate_code(self, ast: ModuleDeclaration, target_language: str) -> Dict[str, str]:
		"""Generate code for target language"""
		try:
			return self.code_generator.generate(ast, target_language)
		except Exception as e:
			self.logger.error(f"Code generation failed: {e}")
			raise
	
	def _write_output_files(self, generated_files: Dict[str, str], output_dir: Path):
		"""Write generated files to disk"""
		output_dir.mkdir(parents=True, exist_ok=True)
		
		for filename, content in generated_files.items():
			file_path = output_dir / filename
			
			# Create subdirectories if needed
			file_path.parent.mkdir(parents=True, exist_ok=True)
			
			with open(file_path, 'w', encoding='utf-8') as f:
				f.write(content)
			
			self.logger.debug(f"Generated: {file_path}")
	
	# ========================================
	# Utility Methods
	# ========================================
	
	def get_supported_targets(self) -> List[str]:
		"""Get list of supported target languages/frameworks"""
		return ['python', 'flask-appbuilder']
	
	def validate_source(self, source_code: str) -> List[Union[APGSyntaxError, SemanticError]]:
		"""Validate APG source code without generating output"""
		result = self.compile_string(source_code, "validation", "python")
		return result.errors + result.warnings
	
	def get_compilation_info(self) -> Dict[str, Any]:
		"""Get information about the last compilation"""
		if not self.last_result:
			return {"status": "no_compilation"}
		
		return {
			"status": "success" if self.last_result.success else "failed",
			"target_language": self.last_result.target_language,
			"files_generated": len(self.last_result.generated_files),
			"errors": len(self.last_result.errors),
			"warnings": len(self.last_result.warnings),
			"compilation_time": self.last_result.compilation_time,
			"module_name": self.last_result.module.name if self.last_result.module else None
		}
	
	def set_config(self, **kwargs):
		"""Update compiler configuration"""
		for key, value in kwargs.items():
			if hasattr(self.config, key):
				setattr(self.config, key, value)
			else:
				self.logger.warning(f"Unknown config option: {key}")


# ========================================
# Convenience Functions
# ========================================

def compile_apg_file(source_file: Union[str, Path], 
					output_dir: Optional[Union[str, Path]] = None,
					target: str = "flask-appbuilder") -> CompilationResult:
	"""
	Convenience function to compile a single APG file.
	
	Args:
		source_file: Path to APG source file
		output_dir: Output directory (optional)
		target: Target framework
		
	Returns:
		CompilationResult
	"""
	config = CodeGenConfig(target_language=target)
	compiler = APGCompiler(config)
	return compiler.compile_file(source_file, output_dir, target)


def compile_apg_string(source_code: str, 
					  target: str = "flask-appbuilder") -> CompilationResult:
	"""
	Convenience function to compile APG source code from string.
	
	Args:
		source_code: APG source code
		target: Target framework
		
	Returns:
		CompilationResult
	"""
	config = CodeGenConfig(target_language=target)
	compiler = APGCompiler(config)
	return compiler.compile_string(source_code, "main", target)


def validate_apg_syntax(source_code: str) -> List[Union[APGSyntaxError, SemanticError]]:
	"""
	Convenience function to validate APG syntax without code generation.
	
	Args:
		source_code: APG source code to validate
		
	Returns:
		List of syntax and semantic errors
	"""
	compiler = APGCompiler()
	return compiler.validate_source(source_code)


def test_compiler():
	"""Test the APG compiler with sample code"""
	test_code = '''
	module test_app version 1.0.0 {
		description: "Test APG application";
		author: "APG Compiler";
	}
	
	agent TestAgent {
		name: str = "Test Agent";
		status: str = "idle";
		
		process: () -> str = {
			return "Hello from APG!";
		};
		
		get_status: () -> str = {
			return status;
		};
	}
	'''
	
	print("Testing APG Compiler...")
	result = compile_apg_string(test_code)
	result.print_summary()
	
	if result.success:
		print("\nGenerated files:")
		for filename in result.generated_files.keys():
			print(f"  - {filename}")
	
	return result


if __name__ == "__main__":
	test_compiler()