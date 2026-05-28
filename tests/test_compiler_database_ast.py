"""Source-backed database AST coverage for executable APG compilation."""

from compiler.ast_builder import ASTBuilder, DatabaseDeclaration
from compiler.parser import APGParser
from compiler.semantic_analyzer import SemanticAnalyzer


DATABASE_SOURCE = """
module data_ops version 1.0.0 {
    description: "Database AST test";
}

db LedgerDB {
    url: "postgresql://localhost:5432/ledger";
    host: "localhost";
    port: 5432;
    database: "ledger";
    ssl: true;

    schema accounting {
        table journals {
            id serial [pk]
            external_id varchar(64) [unique, not null]
            amount decimal(12,2) [not null, default: 0]
            created_at timestamp [default: now()]

            indexes {
                (external_id) [unique, name: "idx_journals_external_id"]
                (created_at) [type: btree]
            }
        }
    }
}
"""


def test_source_database_builds_typed_ast_with_config_schema_and_indexes():
	parse_result = APGParser().parse_string(DATABASE_SOURCE, "database.apg")
	assert parse_result["success"] is True, parse_result["errors"]

	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "database.apg")
	database = ast.entities[0]

	assert isinstance(database, DatabaseDeclaration)
	assert database.connection_config == {
		"url": "postgresql://localhost:5432/ledger",
		"host": "localhost",
		"port": 5432,
		"database": "ledger",
		"ssl": True,
	}
	assert [property.name for property in database.properties] == [
		"url",
		"host",
		"port",
		"database",
		"ssl",
	]

	schema = database.schemas[0]
	table = schema.tables[0]
	assert schema.name == "accounting"
	assert table.name == "journals"
	assert [column.name for column in table.columns] == [
		"id",
		"external_id",
		"amount",
		"created_at",
	]
	assert table.columns[0].is_primary_key is True
	assert table.columns[1].is_nullable is False
	assert table.columns[2].default_value == "0"
	assert table.indexes[0].name == "idx_journals_external_id"
	assert table.indexes[0].columns == ["external_id"]
	assert table.indexes[0].is_unique is True
	assert table.indexes[1].index_type == "btree"


def test_database_semantic_validation_uses_connection_config():
	parse_result = APGParser().parse_string(DATABASE_SOURCE, "database.apg")
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "database.apg")

	result = SemanticAnalyzer().analyze(ast)

	assert result["success"] is True
	assert not [
		warning for warning in result["warnings"]
		if "connection configuration" in str(warning)
	]
