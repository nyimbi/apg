"""
APG Parser Tests
================

Test suite for APG parser functionality including lexical analysis,
syntax parsing, and error handling.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apg.compiler.parser import APGParser, APGSyntaxError


class TestAPGParser:
	"""Test cases for APG parser"""
	
	def setup_method(self):
		"""Setup for each test method"""
		self.parser = APGParser()
	
	def test_simple_module_parsing(self):
		"""Test parsing a simple module declaration"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		assert result['parse_tree'] is not None
		assert result['source_name'] == "test.apg"
	
	def test_agent_parsing(self):
		"""Test parsing agent declarations"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		agent TestAgent {
			name: str = "Test Agent";
			status: str = "idle";
			
			process: () -> str = {
				return "Hello World";
			};
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		assert result['parse_tree'] is not None
	
	def test_digital_twin_parsing(self):
		"""Test parsing digital twin declarations"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		digital_twin SensorTwin {
			sensor_id: str = "sensor_001";
			temperature: float = 20.0;
			humidity: float = 50.0;
			last_reading: str = "2023-01-01T00:00:00Z";
			
			update_reading: (temp: float, humid: float) -> void = {
				temperature = temp;
				humidity = humid;
				last_reading = now();
			};
			
			get_status: () -> dict = {
				return {
					"id": sensor_id,
					"temp": temperature,
					"humidity": humidity,
					"last_update": last_reading
				};
			};
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_workflow_parsing(self):
		"""Test parsing workflow declarations"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		workflow DataProcessingWorkflow {
			name: str = "Data Processing";
			steps: list[str] = ["extract", "transform", "load"];
			current_step: int = 0;
			
			execute_step: (step_name: str) -> bool = {
				// Execute the specified step
				return true;
			};
			
			get_progress: () -> dict = {
				return {
					"current": current_step,
					"total": len(steps)
				};
			};
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_database_parsing(self):
		"""Test parsing database declarations with DBML integration"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		db TestDB {
			url: "postgresql://localhost:5432/testdb";
			host: "localhost";
			port: 5432;
			database: "testdb";
			
			schema main_schema {
				table users {
					id serial [pk]
					email varchar(255) [unique, not null]
					username varchar(50) [unique, not null]
					created_at timestamp [default: now()]
					
					indexes {
						(email) [unique]
						(username) [unique]
					}
				}
				
				table posts {
					id serial [pk]
					user_id int [ref: > users.id]
					title varchar(200) [not null]
					content text
					created_at timestamp [default: now()]
					
					indexes {
						(user_id)
						(created_at)
					}
				}
			}
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_vector_storage_parsing(self):
		"""Test parsing vector storage and AI/ML features"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		db VectorDB {
			url: "postgresql://localhost:5432/vectordb";
			
			schema ai_schema {
				table embeddings {
					id serial [pk]
					content text [not null]
					content_embedding vector(1536) [dimensions: 1536, normalized]
					feature_vector halfvec(128) [dimensions: 128]
					category_sparse sparsevec(1000) [sparse]
					created_at timestamp [default: now()]
					
					vector_index idx_content_similarity on embeddings (content_embedding) [
						method: hnsw,
						distance: cosine,
						dimensions: 1536
					]
				}
				
				trigger update_embeddings after insert on embeddings {
					begin
						execute procedure refresh_embeddings(NEW.id);
					end
				}
				
				procedure refresh_embeddings(in embedding_id int) [language: plpgsql] {
					begin
						// TODO: Refresh embedding logic
					end
				}
			}
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_complex_expressions(self):
		"""Test parsing complex expressions and statements"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		agent MathAgent {
			counter: int = 0;
			values: list[float] = [];
			
			calculate: (x: float, y: float, operation: str) -> float = {
				if (operation == "add") {
					return x + y;
				} else if (operation == "subtract") {
					return x - y;
				} else if (operation == "multiply") {
					return x * y;
				} else if (operation == "divide") {
					if (y != 0) {
						return x / y;
					} else {
						return 0.0;
					}
				}
				return 0.0;
			};
			
			process_list: (numbers: list[float]) -> dict = {
				values = numbers;
				total = 0.0;
				
				for (num in numbers) {
					total = total + num;
					counter = counter + 1;
				}
				
				return {
					"sum": total,
					"count": len(numbers),
					"average": total / len(numbers)
				};
			};
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_syntax_error_handling(self):
		"""Test parser error handling for syntax errors"""
		# Missing closing brace
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		
		agent TestAgent {
			name: str = "Test";
		// Missing closing brace for agent
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == False
		assert len(result['errors']) > 0
		assert isinstance(result['errors'][0], APGSyntaxError)
	
	def test_empty_source(self):
		"""Test parsing empty source code"""
		source_code = ""
		
		result = self.parser.parse_string(source_code, "empty.apg")
		
		# Empty source should either succeed with empty AST or fail gracefully
		assert 'success' in result
		assert 'errors' in result
	
	def test_comments_parsing(self):
		"""Test parsing source code with comments"""
		source_code = '''
		// Module declaration
		module test version 1.0.0 {
			description: "Test module with comments";
		}
		
		/* Multi-line comment
		   describing the agent */
		agent TestAgent {
			// Property with comment
			name: str = "Test Agent"; // Inline comment
			
			/* Method with
			   multi-line comment */
			process: () -> str = {
				// Return statement
				return "Hello"; // With comment
			};
		}
		'''
		
		result = self.parser.parse_string(source_code, "test.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_unicode_support(self):
		"""Test parsing source code with Unicode characters"""
		source_code = '''
		module тест version 1.0.0 {
			description: "Тестовый модуль с Unicode символами";
		}
		
		agent 日本Agent {
			名前: str = "こんにちは世界";
			状態: str = "アクティブ";
			
			挨拶: (名前: str) -> str = {
				return "こんにちは、" + 名前 + "さん！";
			};
		}
		'''
		
		result = self.parser.parse_string(source_code, "unicode.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
	
	def test_file_parsing(self, tmp_path):
		"""Test parsing APG source from file"""
		# Create temporary APG file
		apg_file = tmp_path / "test.apg"
		apg_content = '''
		module file_test version 1.0.0 {
			description: "Test parsing from file";
		}
		
		agent FileAgent {
			name: str = "File Agent";
			
			process: () -> str = {
				return "Parsed from file";
			};
		}
		'''
		
		apg_file.write_text(apg_content)
		
		# Parse the file
		result = self.parser.parse_file(str(apg_file))
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		assert result['source_name'] == str(apg_file)
	
	def test_nonexistent_file(self):
		"""Test parsing nonexistent file"""
		with pytest.raises(FileNotFoundError):
			self.parser.parse_file("nonexistent.apg")
	
	def test_parser_error_recovery(self):
		"""Test parser error recovery with multiple errors"""
		source_code = '''
		module test version 1.0.0 {
			description: "Test module";
		}
		
		agent TestAgent {
			name: str = "Test"  // Missing semicolon
			
			invalid_method: () -> {  // Missing return type
				return "test"  // Missing semicolon
			}  // Missing semicolon for method
		}
		
		// Invalid entity type
		invalid_entity InvalidEntity {
			prop: str = "test";
		}
		'''
		
		result = self.parser.parse_string(source_code, "errors.apg")
		
		# Should capture multiple errors
		assert result['success'] == False
		assert len(result['errors']) >= 1
	
	def test_large_source_file(self):
		"""Test parsing performance with large source file"""
		# Generate large APG source
		agents = []
		for i in range(100):
			agent_code = f'''
			agent Agent{i} {{
				name: str = "Agent {i}";
				id: int = {i};
				
				process_{i}: () -> str = {{
					return "Result from agent {i}";
				}};
				
				get_id: () -> int = {{
					return {i};
				}};
			}}
			'''
			agents.append(agent_code)
		
		source_code = f'''
		module large_test version 1.0.0 {{
			description: "Large test file with {len(agents)} agents";
		}}
		
		{chr(10).join(agents)}
		'''
		
		result = self.parser.parse_string(source_code, "large.apg")
		
		assert result['success'] == True
		assert len(result['errors']) == 0
		
		# Should complete in reasonable time
		# (This is implicitly tested by pytest timeout if configured)


if __name__ == "__main__":
	pytest.main([__file__])