from typing import List

class SQLGeneratorVisitor:
    def __init__(self):
        self.sql = ""

    # Helper functions for generating SQL
    def create_table(self, table_name: str, columns: List[str], primary_key: str) -> None:
        self.sql += f"CREATE TABLE {table_name} (\n"
        self.sql += ",\n".join(columns)
        self.sql += f",\nPRIMARY KEY ({primary_key})" if primary_key else ""
        self.sql += "\n);\n\n"

    def create_index(self, table_name: str, index_name: str, columns: List[str]) -> None:
        self.sql += f"CREATE INDEX {index_name} ON {table_name} ({', '.join(columns)});\n"

    # Visitor methods for each rule in the grammar
    def visit_appgen(self, node) -> None:
        for child in node.children:
            self.visit(child)

    def visit_table(self, node) -> None:
        table_name = node.table_name.text
        mixin_names = [mixin.text for mixin in node.mixin_list.children] if node.mixin_list else []
        columns = []
        primary_key_column = None

        for column_node in node.column_list.children:
            column_name = column_node.column_name.text
            column_type = column_node.column_type.text.upper()
            options = [opt.text.upper() for opt in column_node.column_option_list.children]
            column_declaration = f"{column_name} {column_type}"

            if "PRIMARY KEY" in options:
                primary_key_column = column_name

            if "NOT NULL" in options:
                column_declaration += " NOT NULL"

            if "UNIQUE" in options:
                column_declaration += " UNIQUE"

            if "REFERENCES" in options:
                ref_table, ref_column = options[options.index("REFERENCES")+1:options.index("REFERENCES")+3]
                column_declaration += f" REFERENCES {ref_table}({ref_column})"

            if "DEFAULT" in options:
                default_value = options[options.index("DEFAULT")+1]
                column_declaration += f" DEFAULT {default_value}"

            columns.append(column_declaration)

        self.create_table(table_name, columns, primary_key_column)

        for index_node in node.index_spec.children:
            index_name = index_node.index_name.text
            index_columns = [col.text for col in index_node.children[1].children]
            self.create_index(table_name, index_name, index_columns)

    def visit_object(self, node) -> None:
        for child in node.children:
            self.visit(child)

    def visit_ident(self, node) -> None:
        pass  # Identifiers are ignored in SQL generation

    # Visit mixins in a table definition
    def visit_mixin(self, node) -> None:
        pass  # Mixins are ignored in SQL generation

    # Visit enum in a table definition
    def visit_enum(self, node) -> None:
        enum_name = node.enum_name.text
        enum_items = [enum_item.text.split("[")[0] for enum_item in node.enum_list.children]
        self.sql += f"CREATE TYPE {enum_name} AS ENUM ({', '.join(enum_items)});\n\n"

    # Visit external references in a table definition
    def visit_ext_ref(self, node) -> None:
        pass  # External references are ignored in SQL generation

    # Visit indexes in a table definition
    def visit_indexes(self, node) -> None:
        pass  # Indexes were handled in the visit_table method

    # Visit column options
    def visit_column_option(self, node) -> None:
        pass  # Column options are handled in the visit_table method

    # Visit enum item in an enum definition
    def visit_enum_item(self, node) -> None:
        pass  # Enum items are handled in the visit_enum method

    # Visit enum value in an enum definition
    def visit_enum_value(self, node) -> None:
        pass  # Enum values are handled in the visit_enum method

    def visit_primary_key(self, node) -> None:
        pass  # Primary keys are handled in the visit_table method

    # Visit attributes of the project
    def visit_project_property(self, node) -> None:
        pass  # Project properties are ignored in SQL generation