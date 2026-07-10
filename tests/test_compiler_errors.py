from compiler.compiler import APGCompiler
from compiler.semantic_analyzer import SemanticError


def _compile(source: str):
	return APGCompiler().compile_string(source, "diagnostics.apg")


def _diagnostic_text(diagnostics) -> str:
	return "\n".join(str(diagnostic) for diagnostic in diagnostics)


def test_error_includes_line_number():
	result = _compile(
		"""module diagnostics version 1.0.0 { }
table Foo {
	x: integer;
}
"""
	)

	assert result.success is False
	assert "line " in str(result.errors[0])
	assert "col " in str(result.errors[0])


def test_did_you_mean_for_typo():
	result = _compile(
		"""module diagnostics version 1.0.0 { }
table Foo {
	x: integer;
}
"""
	)

	assert result.success is False
	message = _diagnostic_text(result.errors)
	assert "Unknown type 'integer'" in message
	assert "Did you mean: int?" in message


def test_duplicate_entity_name_error():
	result = _compile(
		"""module diagnostics version 1.0.0 { }
table Foo {
	x: str;
}
table Foo {
	y: str;
}
"""
	)

	assert result.success is False
	assert "Duplicate entity name: Foo" in _diagnostic_text(result.errors)


def test_duplicate_field_name_error():
	result = _compile(
		"""module diagnostics version 1.0.0 { }
table Foo {
	x: str;
	x: int;
}
"""
	)

	assert result.success is False
	assert "Duplicate field name: x in entity Foo" in _diagnostic_text(result.errors)


def test_empty_entity_warning():
	result = _compile(
		"""module diagnostics version 1.0.0 { }
table Foo {
}
"""
	)

	assert result.success is True
	assert any(
		isinstance(warning, SemanticError)
		and warning.error_type == "warning"
		and "Entity Foo has no fields" in str(warning)
		for warning in result.warnings
	)
