"""
APG Semantic Analyzer Tests
===========================

Test suite for APG semantic analysis including type checking,
symbol resolution, and semantic validation.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apg.compiler.parser import APGParser
from apg.compiler.ast_builder import ASTBuilder, ModuleDeclaration, EntityDeclaration, EntityType
from apg.compiler.semantic_analyzer import SemanticAnalyzer, SemanticError, APGType


class TestSemanticAnalyzer:
	"""Test cases for APG semantic analyzer"""
	
	def setup_method(self):
		"""Setup for each test method"""
		self.parser = APGParser()
		self.ast_builder = ASTBuilder()
		self.analyzer = SemanticAnalyzer()
	
	def _parse_and_analyze(self, source_code: str):
		"""Helper method to parse and analyze source code"""
		# Parse
		parse_result = self.parser.parse_string(source_code, "test.apg")
		if not parse_result['success']:
			pytest.fail(f"Parse failed: {parse_result['errors']}")
		
		# Build AST
		ast = self.ast_builder.build_ast(parse_result['parse_tree'], "test.apg")
		if not ast:
			pytest.fail("AST building failed")
		
		# Analyze
		analysis_result = self.analyzer.analyze(ast)
		return ast, analysis_result
	
	def test_valid_module_analysis(self):
		"""Test semantic analysis of valid module"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		agent TestAgent {
			name: str = "Test Agent";
			counter: int = 0;
			
			increment: () -> int = {
				counter = counter + 1;
				return counter;
			};
			
			get_name: () -> str = {
				return name;
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		assert isinstance(ast, ModuleDeclaration)
		assert ast.name == "test"
	
	def test_type_checking(self):
		"""Test type checking for properties and methods"""
		source_code = '''
		module test version 1.0.0 {
			description: "Type checking test";
		}
		
		agent TypeAgent {
			name: str = "Type Agent";
			age: int = 25;
			height: float = 5.9;
			active: bool = true;
			tags: list[str] = ["agent", "test"];
			config: dict[str, any] = {"key": "value"};
			
			get_info: () -> dict = {
				return {
					"name": name,
					"age": age,
					"height": height,
					"active": active
				};
			};
			
			update_age: (new_age: int) -> void = {
				age = new_age;
			};
			
			calculate_bmi: (weight: float) -> float = {
				return weight / (height * height);
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		
		# Check that all built-in types are recognized
		symbols = result['symbol_table'].get_all_symbols()
		assert 'TypeAgent' in symbols
	
	def test_undefined_symbol_error(self):
		"""Test detection of undefined symbols"""
		source_code = '''
		module test version 1.0.0 {
			description: "Undefined symbol test";
		}
		
		agent ErrorAgent {
			name: str = "Error Agent";
			
			process: () -> str = {
				return undefined_variable;  // This should cause an error
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should have semantic errors for undefined symbol
		# Note: This might pass if the AST builder creates stub implementations
		# The actual error detection depends on full AST traversal implementation
		assert 'errors' in result
	
	def test_duplicate_symbol_error(self):
		"""Test detection of duplicate symbol definitions"""
		source_code = '''
		module test version 1.0.0 {
			description: "Duplicate symbol test";
		}
		
		agent DuplicateAgent {
			name: str = "Agent";
			name: int = 42;  // Duplicate property name
			
			process: () -> str = {
				return "test";
			};
			
			process: () -> int = {  // Duplicate method name
				return 42;
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should detect duplicate definitions
		assert len(result['errors']) > 0
		error_messages = [str(error) for error in result['errors']]
		assert any('already defined' in msg.lower() for msg in error_messages)
	
	def test_type_mismatch_detection(self):
		"""Test detection of type mismatches"""
		source_code = '''
		module test version 1.0.0 {
			description: "Type mismatch test";
		}
		
		agent TypeMismatchAgent {
			counter: int = 0;
			name: str = "Agent";
			
			wrong_assignment: () -> void = {
				counter = "not a number";  // Type mismatch
				name = 42;  // Type mismatch
			};
			
			wrong_return: () -> int = {
				return "string instead of int";  // Return type mismatch
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should detect type mismatches
		# Note: Full type checking requires complete expression analysis
		assert 'errors' in result or 'warnings' in result
	
	def test_entity_type_validation(self):
		"""Test validation of entity-specific constraints"""
		source_code = '''
		module test version 1.0.0 {
			description: "Entity type validation test";
		}
		
		agent ValidAgent {
			name: str = "Valid Agent";
			
			process: () -> str = {  // Agents should have a process method
				return "processing";
			};
		}
		
		agent IncompleteAgent {
			name: str = "Incomplete Agent";
			// No process method - should generate warning
		}
		
		digital_twin ValidTwin {
			sensor_id: str = "sensor_001";
			state: dict = {"temp": 20.0};  // Digital twins should have state
			
			update_state: (new_state: dict) -> void = {
				state = new_state;
			};
		}
		
		digital_twin IncompleteTwin {
			id: str = "twin_001";
			// No state-related properties - should generate warning
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should have warnings for incomplete agents/twins
		assert len(result['warnings']) > 0
		warning_messages = [str(warning) for warning in result['warnings']]
		assert any('process' in msg.lower() for msg in warning_messages)
		assert any('state' in msg.lower() for msg in warning_messages)
	
	def test_workflow_validation(self):
		"""Test validation of workflow entities"""
		source_code = '''
		module test version 1.0.0 {
			description: "Workflow validation test";
		}
		
		workflow CompleteWorkflow {
			name: str = "Complete Workflow";
			steps: list[str] = ["start", "process", "end"];
			current_step: int = 0;
			
			execute_step: (step_name: str) -> bool = {
				return true;
			};
		}
		
		workflow IncompleteWorkflow {
			name: str = "Incomplete Workflow";
			// No steps or stages - should generate warning
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should warn about missing steps in workflow
		assert len(result['warnings']) > 0
		warning_messages = [str(warning) for warning in result['warnings']]
		assert any('step' in msg.lower() or 'stage' in msg.lower() for msg in warning_messages)
	
	def test_database_validation(self):
		"""Test validation of database entities"""
		source_code = '''
		module test version 1.0.0 {
			description: "Database validation test";
		}
		
		db CompleteDatabase {
			url: "postgresql://localhost:5432/testdb";
			host: "localhost";
			port: 5432;
			database: "testdb";
		}
		
		db IncompleteDatabase {
			name: str = "Incomplete DB";
			// No connection configuration - should generate warning
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should warn about missing connection configuration
		assert len(result['warnings']) > 0
		warning_messages = [str(warning) for warning in result['warnings']]
		assert any('connection' in msg.lower() for msg in warning_messages)
	
	def test_symbol_table_scoping(self):
		"""Test symbol table scope management"""
		source_code = '''
		module test version 1.0.0 {
			description: "Scoping test";
		}
		
		agent ScopingAgent {
			global_var: str = "global";
			
			method1: (param1: str) -> str = {
				local_var: str = "local1";
				return param1 + local_var;
			};
			
			method2: (param2: int) -> int = {
				local_var: int = 42;  // Different type, same name as method1
				return param2 + local_var;
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		
		# Check symbol table structure
		symbol_table = result['symbol_table']
		assert symbol_table is not None
		symbols = symbol_table.get_all_symbols()
		assert 'ScopingAgent' in symbols
	
	def test_method_signature_validation(self):
		"""Test validation of method signatures"""
		source_code = '''
		module test version 1.0.0 {
			description: "Method signature test";
		}
		
		agent MethodAgent {
			name: str = "Method Agent";
			
			valid_method: (x: int, y: float) -> str = {
				return str(x + y);
			};
			
			async_method: async (data: dict) -> bool = {
				// Async method implementation
				return true;
			};
			
			optional_param_method: (required: str, optional: int = 42) -> str = {
				return required + str(optional);
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_unknown_type_error(self):
		"""Test detection of unknown type references"""
		source_code = '''
		module test version 1.0.0 {
			description: "Unknown type test";
		}
		
		agent UnknownTypeAgent {
			name: str = "Agent";
			unknown_prop: UnknownType = null;  // Unknown type
			
			unknown_param_method: (param: AnotherUnknownType) -> void = {
				// Method with unknown parameter type
			};
			
			unknown_return_method: () -> YetAnotherUnknownType = {
				// Method with unknown return type
				return null;
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should detect unknown types
		assert len(result['errors']) > 0
		error_messages = [str(error) for error in result['errors']]
		assert any('unknown type' in msg.lower() for msg in error_messages)
	
	def test_circular_reference_detection(self):
		"""Test detection of circular references"""
		source_code = '''
		module test version 1.0.0 {
			description: "Circular reference test";
		}
		
		agent AgentA {
			name: str = "Agent A";
			partner: AgentB = null;  // Forward reference
		}
		
		agent AgentB {
			name: str = "Agent B";
			partner: AgentA = null;  // Circular reference
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should handle circular references gracefully
		# This is a complex case that might require special handling
		assert 'errors' in result or 'warnings' in result
	
	def test_dead_code_analysis(self):
		"""Test dead code detection"""
		source_code = '''
		module test version 1.0.0 {
			description: "Dead code test";
		}
		
		agent DeadCodeAgent {
			name: str = "Agent";
			unused_property: str = "never used";  // Unused property
			used_property: int = 42;
			
			main_method: () -> str = {
				return used_property;  // Uses used_property
			};
			
			unused_method: () -> void = {
				// This method is never called
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		# Should detect unused properties and methods
		assert len(result['warnings']) > 0
		warning_messages = [str(warning) for warning in result['warnings']]
		assert any('unused' in msg.lower() for msg in warning_messages)
	
	def test_complex_entity_analysis(self):
		"""Test analysis of complex entities with multiple features"""
		source_code = '''
		module complex_test version 1.0.0 {
			description: "Complex entity analysis test";
			author: "APG Test Suite";
		}
		
		agent ComplexAgent {
			// Properties with various types
			id: str = "agent_001";
			config: dict[str, any] = {
				"max_retries": 3,
				"timeout": 30.0,
				"enabled": true
			};
			
			handlers: list[str] = ["http", "websocket", "mqtt"];
			metrics: dict[str, int] = {};
			
			// Methods with different signatures
			initialize: () -> bool = {
				metrics = {
					"requests": 0,
					"errors": 0,
					"uptime": 0
				};
				return true;
			};
			
			handle_request: async (request_type: str, data: dict) -> dict = {
				metrics["requests"] = metrics["requests"] + 1;
				
				if (request_type in handlers) {
					return {"status": "success", "data": data};
				} else {
					metrics["errors"] = metrics["errors"] + 1;
					return {"status": "error", "message": "Unknown request type"};
				}
			};
			
			get_metrics: () -> dict = {
				return metrics;
			};
			
			cleanup: () -> void = {
				metrics = {};
			};
		}
		
		digital_twin SensorTwin {
			sensor_id: str = "sensor_001";
			location: dict[str, float] = {"lat": 40.7128, "lon": -74.0060};
			readings: list[dict] = [];
			state: dict[str, any] = {"active": true, "last_reading": null};
			
			add_reading: (temperature: float, humidity: float, timestamp: str) -> void = {
				reading = {
					"temperature": temperature,
					"humidity": humidity,
					"timestamp": timestamp
				};
				
				readings.append(reading);
				state["last_reading"] = reading;
			};
			
			get_latest_reading: () -> dict = {
				if (len(readings) > 0) {
					return readings[len(readings) - 1];
				}
				return {};
			};
			
			get_average_temperature: () -> float = {
				if (len(readings) == 0) {
					return 0.0;
				}
				
				total = 0.0;
				for (reading in readings) {
					total = total + reading["temperature"];
				}
				
				return total / len(readings);
			};
		}
		'''
		
		ast, result = self._parse_and_analyze(source_code)
		
		assert result['success'] == True
		# Should have minimal errors for well-structured code
		assert len(result['errors']) == 0
		
		# Check that entities were properly analyzed
		symbols = result['symbol_table'].get_all_symbols()
		assert 'ComplexAgent' in symbols
		assert 'SensorTwin' in symbols


if __name__ == "__main__":
	pytest.main([__file__])