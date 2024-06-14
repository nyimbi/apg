from typing import List
from AppGenVisitor import AppGenVisitor

class PostgresSQLGenerator:
    def __init__(self):
        self.sql = ''
        self.table_names = []

    def generate(self, ctx) -> str:
        self.visitAppgen(ctx)
        return self.sql

    def visitAppgen(self, ctx):
        for child_ctx in ctx.getChildren():
            self.visit(child_ctx)

    def visitTable(self, ctx):
        table_name = ctx.tableName().getText()
        column_defs = []
        for column_ctx in ctx.column():
            column_name = column_ctx.ID().getText()
            column_type = self.visit(column_ctx.type())
            column_property_str = ""
            for prop_ctx in column_ctx.property():
                property_name = prop_ctx.getText()
                if prop_ctx.STRING() is not None:
                    property_value = prop_ctx.STRING().getText().strip('"')
                else:
                    property_value = None

                if property_name in ['nullable', 'required']:
                    column_property_str += f'{property_name} {not property_value}, '
                elif property_value is not None:
                    column_property_str += f"{property_name}='{property_value}', "
                else:
                    column_property_str += f"{property_name}, "

            if column_ctx.type().getText() == 'serial':
                column_defs.append(f"{column_name} SERIAL PRIMARY KEY")
            else:
                column_defs.append(f"{column_name} {column_type} {column_property_str}")

        mixin_strs = []
        for mixin_ctx in ctx.mixin():
            # Instead of creating a separate table for mixins, we just add the columns to the current table
            for column_ctx in mixin_ctx.column():
                column_name = column_ctx.ID().getText()
                column_type = self.visit(column_ctx.type())
                column_property_str = ""
                for prop_ctx in column_ctx.property():
                    property_name = prop_ctx.getText()
                    if prop_ctx.STRING() is not None:
                        property_value = prop_ctx.STRING().getText().strip('"')
                    else:
                        property_value = None

                    if property_name in ['nullable', 'required']:
                        column_property_str += f'{property_name} {not property_value}, '
                    elif property_value is not None:
                        column_property_str += f"{property_name}='{property_value}', "
                    else:
                        column_property_str += f"{property_name}, "

                if column_ctx.type().getText() == 'serial':
                    column_defs.append(f"{column_name} SERIAL PRIMARY KEY")
                else:
                    column_defs.append(f"{column_name} {column_type} {column_property_str}")

        self.table_names.append(table_name)
        self.sql += f"CREATE TABLE {table_name} (\n"
        self.sql += ",\n".join(column_defs)
        self.sql += ");\n"

    def visitEnum(self, ctx):
        enum_name = ctx.enumName().getText()
        enum_values = [v.getText() for v in ctx.getChild(3).getChildren() if v.ID() is not None]

        self.sql += f"CREATE TYPE {enum_name} AS ENUM ({', '.join([f'\'{v}\'' for v in enum_values])});\n"

    def visitID(self, ctx) -> str:
        # Ref ID, display/show, and widget/control might contain a dot
        # If the ID is NOT in the table names list, then it must be an enum name or a property name rather than a table name
        if ctx.getText() not in self.table_names:
            return 'VARCHAR'
        return ctx.getText().lower()

    def visitVarchar(self, ctx) -> str:
        return f"VARCHAR({ctx.INT().getText()})"

    def visitType(self, ctx) -> str:
        if ctx.getChildCount() > 1:
            if ctx.getChild(1).getText() == '?':
                return "VARCHAR"

        type_str = ctx.getText().lower()
        if type_str == "ref":
            next_ctx = ctx.parentCtx.getChild(ctx.getChildIndex() + 1)
            next_text = next_ctx.getText().lower()
            if next_text == "manytomany":
                # Create the association table name by concatenating the table names
                # with an underscore in-between, and sorting them alphabetically
                table_names = sorted([ctx.parentCtx.parentCtx.getChild(i).getText()
                                      for i in range(ctx.parentCtx.parentCtx.getChildCount())
                                      if
                                      ctx.parentCtx.parentCtx.getChild(i).getRuleIndex() == modelParser.RULE_className])
                association_table_name = "_".join(table_names)

                # Add ID columns for each of the tables involved
                schema = f"CREATE TABLE {association_table_name} (\n"
                for i in range(2):
                    schema += f"\t{table_names[i]}_id INT,\n"
                schema += "\tid INT,\n"
                schema += f"\tPRIMARY KEY ({table_names[0]}_id, {table_names[1]}_id)\n)"
                return schema

        elif type_str in ['file', 'image']:
            return 'VARCHAR(160)'
        else:
            return type_str.upper()