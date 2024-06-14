class FABModelsVisitor:
    def __init__(self):
        self.models = []

    # Visitor methods for each rule in the grammar
    def visit_appgen(self, node):
        for child in node.children:
            self.visit(child)

    def visit_table(self, node):
        table_name = node.table_name.text
        schema = ""

        columns = []
        for column_node in node.column_list.children:
            column_name = column_node.column_name.text
            column_type = column_node.column_type.text.lower()

            if column_type.startswith("int"):
                column_type = "Integer"
            elif column_type.startswith("float"):
                column_type = "Float"
            elif column_type.startswith("bool"):
                column_type = "Boolean"
            elif column_type.startswith("text"):
                column_type = "Text"
            elif column_type.startswith("string"):
                column_type = "String"
            elif column_type.startswith("datetime"):
                column_type = "DateTime"
            else:
                column_type = "String"

            options = [opt.text.lower() for opt in column_node.column_option_list.children]

            if "primary key" in options:
                primary_key = True
            else:
                primary_key = False

            if "unique" in options:
                unique = True
            else:
                unique = False

            if "nullable" in options:
                nullable = True
            else:
                nullable = False

            if "default" in options:
                default = options[options.index("default") + 1]
            else:
                default = None

            if "foreign key" in options:
                ref_table, ref_column = options[options.index("references") + 1:]
                foreign_key = f"{ref_table}.{ref_column}"
            else:
                foreign_key = None

            columns.append((column_name, column_type, primary_key, unique, nullable, default, foreign_key))

        for mixin_node in node.mixin_list.children:
            mixin_name = mixin_node.text
            module_name = self._get_module_name(mixin_name)
            schema += f"from {module_name} import {mixin_name}\n"
            columns.extend(self._get_mixin_columns(mixin_name))

        self.models.append((table_name, columns, schema))

    def visit_enum(self, node):
        enum_name = node.enum_name.text
        enum_items = [enum_item.text.split("[")[0] for enum_item in node.enum_list.children]
        self.models.append((enum_name, enum_items, ""))

    def visit_mixin(self, node):
        mixin_name = node.text
        module_name = self._get_module_name(mixin_name)
        mixin_class_name = self._get_class_name(mixin_name)
        schema = f"from {module_name} import {mixin_name} as {mixin_class_name}\n"
        columns = self._get_mixin_columns(mixin_name)
        self.models.append((mixin_class_name, columns, schema))

    # Helper methods for generating models
    def _get_module_name(self, model_name):
        return model_name.lower()

    def _get_class_name(self, model_name):
        return model_name.capitalize()

    def _get_mixin_columns(self, mixin_name):
        mixin_module_name = self._get_module_name(mixin_name)
        mixin_class_name = self._get_class_name(mixin_name)
        mixin_class = getattr(__import__(mixin_module_name, fromlist=[mixin_class_name]), mixin_class_name)
        columns = [(field.name, field.type.python_type.__name__.capitalize(), False, False, True, None, None) for field in mixin_class.__table__.columns]
        return columns