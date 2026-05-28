"""Source-backed database AST coverage for executable APG compilation."""

from compiler.ast_builder import ASTBuilder, DatabaseDeclaration
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator
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
            account_id int [ref: > accounts.id, not null]
            amount decimal(12,2) [not null, default: 0]
            created_at timestamp [default: now()]

            indexes {
                (external_id) [unique, name: "idx_journals_external_id"]
                (created_at) [type: btree]
            }
        }

        table accounts {
            id serial [pk]
            code varchar(32) [unique, not null]
        }
    }
}
"""


BROKEN_DATABASE_SOURCE = """
db BrokenDB {
    url: "postgresql://localhost:5432/broken";

    schema accounting {
        table journals {
            id serial [pk]
            account_id int [ref: > accounts.missing_id]
        }

        table accounts {
            id serial [pk]
        }
    }
}
"""


SCHEMA_QUALIFIED_DATABASE_SOURCE = """
db MultiSchemaDB {
    schema sales {
        table orders {
            id serial [pk]
            account_id int [ref: > accounting.accounts.id]
        }
    }

    schema accounting {
        table accounts {
            id serial [pk]
        }
    }
}
"""


AMBIGUOUS_DATABASE_SOURCE = """
db AmbiguousDB {
    schema sales {
        table accounts {
            id serial [pk]
        }
    }

    schema accounting {
        table accounts {
            id serial [pk]
        }
    }

    schema operations {
        table journals {
            id serial [pk]
            account_id int [ref: > accounts.id]
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
		"account_id",
		"amount",
		"created_at",
	]
	assert table.columns[0].is_primary_key is True
	assert table.columns[1].is_nullable is False
	assert table.columns[2].reference == {
		"kind": ">",
		"relationship": "many_to_one",
		"table": "accounts",
		"column": "id",
		"target": "accounts.id",
	}
	assert table.columns[2].is_nullable is False
	assert table.columns[3].default_value == "0"
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


def test_generated_python_metadata_preserves_database_schema_details():
	parse_result = APGParser().parse_string(DATABASE_SOURCE, "database.apg")
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	namespace = {}
	exec(files["app.py"], namespace)

	database = namespace["list_entities"]()[0]

	assert database["connection_config"]["url"] == "postgresql://localhost:5432/ledger"
	assert database["schemas"][0]["name"] == "accounting"
	assert database["schemas"][0]["tables"][0]["name"] == "journals"
	assert database["schemas"][0]["tables"][0]["columns"][0] == {
		"name": "id",
		"type": "serial",
		"primary_key": True,
		"nullable": True,
		"default": None,
		"constraints": ["pk"],
	}
	assert database["schemas"][0]["tables"][0]["columns"][2]["reference"] == {
		"kind": ">",
		"relationship": "many_to_one",
		"table": "accounts",
		"column": "id",
		"target": "accounts.id",
	}
	assert database["schemas"][0]["tables"][0]["indexes"][0] == {
		"name": "idx_journals_external_id",
		"columns": ["external_id"],
		"unique": True,
		"type": None,
	}

	relationship_edges = namespace["relationship_graph"]()["edges"]
	assert {
		"from": "LedgerDB.accounting.journals",
		"to": "LedgerDB.accounting.accounts",
		"field": "account_id",
		"relationship": "many_to_one",
		"target_column": "id",
	} in relationship_edges


def test_generated_python_exposes_database_catalog_routes_and_openapi():
	parse_result = APGParser().parse_string(DATABASE_SOURCE, "database.apg")
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	namespace = {}
	exec(files["app.py"], namespace)

	assert namespace["list_databases"]()[0]["name"] == "LedgerDB"
	assert namespace["describe_application"]()["databases"][0]["name"] == "LedgerDB"

	status, payload = namespace["_route_payload"]("/databases")
	assert status == 200
	assert payload["databases"][0]["schemas"][0]["name"] == "accounting"

	status, payload = namespace["_route_payload"]("/databases/status")
	assert status == 200
	assert payload["valid"] is True
	assert payload["database_count"] == 1
	assert payload["schema_count"] == 1
	assert payload["table_count"] == 2
	assert payload["reference_count"] == 1

	status, payload = namespace["_route_payload"]("/databases/LedgerDB/schemas")
	assert status == 200
	assert payload["schemas"][0]["tables"][0]["name"] == "journals"

	status, payload = namespace["_route_payload"]("/databases/Missing/schemas")
	assert status == 404
	assert payload == {"error": "unknown_database", "database": "Missing"}

	openapi = namespace["openapi_document"]()
	assert "/databases" in openapi["paths"]
	assert "/databases/status" in openapi["paths"]
	assert "/databases/LedgerDB/schemas" in openapi["paths"]
	schemas = openapi["components"]["schemas"]
	assert schemas["DatabaseCatalog"]["properties"]["databases"]["items"] == {
		"$ref": "#/components/schemas/DatabaseCatalogEntry"
	}
	assert schemas["DatabaseColumn"]["properties"]["reference"]["oneOf"][0] == {
		"$ref": "#/components/schemas/DatabaseReference"
	}
	assert openapi["paths"]["/databases"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/DatabaseCatalog"
	}
	assert openapi["paths"]["/databases/status"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/DatabaseStatus"
	}
	assert openapi["paths"]["/databases/LedgerDB/schemas"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/DatabaseSchemaCatalog"
	}

	metrics = namespace["metrics_snapshot"]()
	assert metrics["database_status"]["table_count"] == 2


def test_generated_ui_surfaces_database_catalog_status_and_schema_links():
	parse_result = APGParser().parse_string(DATABASE_SOURCE, "database.apg")
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	namespace = {}
	exec(files["app.py"], namespace)

	status, index_html = namespace["_ui_payload"]("/ui")
	assert status == 200
	assert "/ui/databases" in index_html
	assert "LedgerDB" in index_html

	status, database_html = namespace["_ui_payload"]("/ui/databases")
	assert status == 200
	assert "Status: <strong>valid</strong>" in database_html
	assert "/databases/LedgerDB/schemas" in database_html
	assert "accounting" in database_html
	assert "journals" in database_html
	assert "accounts" in database_html


def test_generated_readme_documents_database_runtime_surface():
	parse_result = APGParser().parse_string(DATABASE_SOURCE, "database.apg")
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	readme = files["README.md"]

	assert "## Databases" in readme
	assert "`GET /databases`" in readme
	assert "`GET /databases/status`" in readme
	assert "`GET /databases/{Database}/schemas`" in readme
	assert "`GET /relationships`" in readme
	assert "`LedgerDB` - 1 schema(s), 2 table(s)" in readme


def test_generated_validation_rejects_broken_database_references():
	parse_result = APGParser().parse_string(BROKEN_DATABASE_SOURCE, "broken_database.apg")
	assert parse_result["success"] is True, parse_result["errors"]
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "broken_database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	namespace = {}
	exec(files["app.py"], namespace)

	validation = namespace["validate_application"]()

	assert validation["valid"] is False
	assert validation["checks"]["database_schemas"]["validated_databases"] == ["BrokenDB"]
	assert validation["errors"] == [
		"database_schemas: BrokenDB.accounting.journals.account_id references unknown column accounts.missing_id"
	]
	assert namespace["self_test"]()["passed"] is False

	status, payload = namespace["_route_payload"]("/databases/status")
	assert status == 422
	assert payload["valid"] is False
	assert payload["validation"]["errors"] == [
		"BrokenDB.accounting.journals.account_id references unknown column accounts.missing_id"
	]


def test_schema_qualified_database_references_validate_and_graph_correctly():
	parse_result = APGParser().parse_string(SCHEMA_QUALIFIED_DATABASE_SOURCE, "multi_schema_database.apg")
	assert parse_result["success"] is True, parse_result["errors"]
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "multi_schema_database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	namespace = {}
	exec(files["app.py"], namespace)

	reference = namespace["list_databases"]()[0]["schemas"][0]["tables"][0]["columns"][1]["reference"]
	assert reference == {
		"kind": ">",
		"relationship": "many_to_one",
		"table": "accounts",
		"column": "id",
		"target": "accounting.accounts.id",
		"schema": "accounting",
	}
	assert namespace["validate_application"]()["valid"] is True
	assert {
		"from": "MultiSchemaDB.sales.orders",
		"to": "MultiSchemaDB.accounting.accounts",
		"field": "account_id",
		"relationship": "many_to_one",
		"target_column": "id",
	} in namespace["relationship_graph"]()["edges"]


def test_generated_validation_rejects_ambiguous_unqualified_database_references():
	parse_result = APGParser().parse_string(AMBIGUOUS_DATABASE_SOURCE, "ambiguous_database.apg")
	assert parse_result["success"] is True, parse_result["errors"]
	ast = ASTBuilder().build_ast(parse_result["parse_tree"], "ambiguous_database.apg")

	files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(ast)
	namespace = {}
	exec(files["app.py"], namespace)

	validation = namespace["validate_application"]()

	assert validation["valid"] is False
	assert validation["errors"] == [
		"database_schemas: AmbiguousDB.operations.journals.account_id references ambiguous table accounts; use schema-qualified target"
	]
