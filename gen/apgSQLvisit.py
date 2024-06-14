from apgVisitor import apgVisitor

class SQLVisitor(apgVisitor):

    def visitStatement(self, ctx):
        return self.visitChildren(ctx)

    def visitDatabase(self, ctx):
        db_name = ctx.dbname().getText()
        db_options = self.visit(ctx.database_options())
        return f"CREATE DATABASE {db_name} {db_options};"

    def visitDatabase_options(self, ctx):
        options = [self.visit(option) for option in ctx.option()]
        return " ".join(options)

    def visitOption(self, ctx):
        identifier = ctx.ident().getText()
        value = self.visit(ctx.getChild(2))
        return f"{identifier} = {value}"

    def visitInt(self, ctx):
        return ctx.INT().getText()

    def visitString(self, ctx):
        return ctx.STRING().getText()

    def visitColumn(self, ctx):
        col_name = ctx.column_name().getText()
        data_type = self.visit(ctx.data_type())
        options = self.visit(ctx.column_option_list()) if ctx.column_option_list() else ""
        return f"{col_name} {data_type} {options}"

    def visitData_type(self, ctx):
        if ctx.enum_name():
            enum_name = ctx.enum_name().getText()
            return f"{enum_name}[]"
        elif ctx.varchar():
            varchar_len = ctx.varchar().getText()
            return f"VARCHAR({varchar_len})"
        elif ctx.INT():
            return "INTEGER"
        elif ctx.BOOL():
            return "BOOLEAN"
        elif ctx.FLOAT():
            return "FLOAT"
        elif ctx.BLOB():
            return "BLOB"
        elif ctx.DATETIME():
            return "DATETIME"
        elif ctx.DECIMAL():
            return "DECIMAL"
        elif ctx.STRING():
            return "STRING"
        else:
            return ""

    def visitColumn_option_list(self, ctx):
        options = [self.visit(option) for option in ctx.column_option()]
        return ", ".join(options)

    def visitPrimary_key(self, ctx):
        return "PRIMARY KEY"

    def visitDefault(self, ctx):
        return f"DEFAULT {self.visit(ctx.column_default())}"

    def visitNullable(self, ctx):
        return "NULL"

    def visitNot_null(self, ctx):
        return "NOT NULL"

    def visitRef_internal(self, ctx):
        ref_name = ctx.ref_name().getText() if ctx.ref_name() else ""
        ref_type = self.visit(ctx.ref_type())
        table_name = self.visit(ctx.table_name())
        col_name = ctx.column_name().getText()
        return f"FOREIGN KEY ({col_name}) REFERENCES {table_name} ({col_name})"

    def visitEnum_internal(self, ctx):
        enum_list = [self.visit(enum_item) for enum_item in ctx.enum_list().enum_item()]
        return f"ENUM({', '.join(enum_list)})"

    def visitEnum_item(self, ctx):
        return ctx.enum_value().getText()

    def visitIndex_ext(self, ctx):
        index_name = ctx.index_name().getText() if ctx.index_name() else ""
        table_name = self.visit(ctx.table_name())
        column_names = ", ".join([c.getText() for c in ctx.column_names().column_name()])
        return f"CREATE INDEX {index_name} ON {table_name} ({column_names})"

    def visitTable(self, ctx):
        table_name = ctx.table_name().getText()
        columns = self.visit(ctx.column_list())
        return f"CREATE TABLE {table_name} ({columns});"

    def visitColumn_list(self, ctx):
        column_defs = [self.visit(column) for column in ctx.column()]
        return ", ".join(column_defs)

    def visitBusiness_rule(self, ctx):
        rule_name = ctx.rule_name().getText()
        condition = self.visit(ctx.condition())
        action = self.visit(ctx.actionExpr())
        return f"CREATE RULE {rule_name} AS ON {condition} DO {action}"

    def visitIfExpr(self, ctx):
        expr = self.visit(ctx.expr())
        return f"WHERE {expr}"

    def visitEveryTimeExpression(self, ctx):
        schedule = self.visit(ctx.schedule())
        return f"EVERY {schedule}"

    def visitOnEventExpression(self, ctx):
        event_desc = self.visit(ctx.event_desc())
        return f"ON EVENT {event_desc}"

    def visitTrigonometricSin(self, ctx):
        arg = self.visit(ctx.expr())
        return f"SIN({arg})"

    # ... more visit methods for other APG grammar rules