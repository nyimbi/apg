from apgListener import apgListener

class SQLGenerator(apgListener):

    def __init__(self):
        self.statements = []

    def get_statements(self):
        return self.statements

    def enterDatabase(self, ctx):
        db_name = ctx.dbname().getText()
        db_options = self.get_db_options(ctx.database_options())
        sql = f"CREATE DATABASE {db_name} {db_options};"
        self.statements.append(sql)

    def get_db_options(self, ctx):
        options = [self.get_option(option) for option in ctx.option()]
        return " ".join(options)

    def get_option(self, ctx):
        identifier = ctx.ident().getText()
        value = self.get_value(ctx.getChild(2))
        return f"{identifier} = {value}"

    def get_value(self, ctx):
        if ctx.STRING():
            return ctx.STRING().getText()
        elif ctx.INT():
            return ctx.INT().getText()
        elif ctx.BOOL():
            return ctx.BOOL().getText()
        elif ctx.FLOAT():
            return ctx.FLOAT().getText()
        elif ctx.column_default():
            return self.get_column_default(ctx.column_default())
        elif ctx.ref_internal():
            return self.get_ref_internal(ctx.ref_internal())
        elif ctx.enum_internal():
            return self.get_enum_internal(ctx.enum_internal())
        else:
            return ""

    def get_column_default(self, ctx):
        value = ctx.getText()
        return value

    def get_ref_internal(self, ctx):
        ref_name = ctx.ref_name().getText() if ctx.ref_name() else ""
        ref_type = self.get_ref_type(ctx.ref_type())
        table_name = self.get_table_name(ctx.table_name())
        col_name = ctx.column_name().getText()
        return f"FOREIGN KEY ({col_name}) REFERENCES {table_name} ({col_name})"

    def get_enum_internal(self, ctx):
        enum_list = [enum_item.enum_value().getText() for enum_item in ctx.enum_list().enum_item()]
        return f"ENUM({', '.join(enum_list)})"

    def get_ref_type(self, ctx):
        if ctx.oneToOne():
            return "ONE-TO-ONE"
        elif ctx.oneToMany():
            return "ONE-TO-MANY"
        elif ctx.manyToOne():
            return "MANY-TO-ONE"
        elif ctx.manyToMany():
            return "MANY-TO-MANY"
        else:
            return ""

    def get_table_name(self, ctx):
        return ctx.getText()

    def enterTable(self, ctx):
        table_name = self.get_table_name(ctx.table_name())
        columns = self.get_column_list(ctx.column_list())
        sql = f"CREATE TABLE {table_name} ({columns});"
        self.statements.append(sql)

    def get_column_list(self, ctx):
        return ", ".join([self.get_column(column) for column in ctx.column()])

    def get_column(self, ctx):
        col_name = ctx.column_name().getText()
        data_type = self.get_data_type(ctx.data_type())
        options = self.get_column_options(ctx.column_option_list()) if ctx.column_option_list() else ""
        return f"{col_name} {data_type} {options}"

    def get_data_type(self, ctx):
        if ctx.enum_name():
            enum_name = ctx.enum_name().getText()
            return f"{enum_name}[]"
        elif ctx.BLOB():
            return "BLOB"
        elif ctx.STRING():
            return "STRING"
        elif ctx.BOOL():
            return "BOOLEAN"
        elif ctx.FLOAT():
            return "FLOAT"
        elif ctx.INT():
            return "INTEGER"
        elif ctx.varchar():
            return f"VARCHAR({ctx.varchar().getText()})"
        else:
            return ""

    def get_column_options(self, ctx):
        return f"[{', '.join([self.get_column_option(option) for option in ctx.column_option()])}]"

    def get_column_option(self, ctx):
        if ctx.primary_key():
            return "PRIMARY KEY"
        elif ctx.DEFAULT():
            return self.get_column_default(ctx.column_default())
        elif ctx.NULLABLE():
            return "NULL"
        elif ctx.NOT_NULL():
            return "NOT NULL"
        elif ctx.unique():
            return "UNIQUE"
        elif ctx.ref_internal():
            return self.get_ref_internal(ctx.ref_internal())
        elif ctx.enum_internal():
            return self.get_enum_internal(ctx.enum_internal())
        else:
            return ""

    def exitBusiness_rule(self, ctx):
        rule_name = ctx.rule_name().getText()
        condition = self.get_condition(ctx.condition())
        action = self.get_action(ctx.actionExpr())
        sql = f"CREATE RULE {rule_name} AS ON {condition} DO {action}"
        self.statements.append(sql)

    def get_condition(self, ctx):
        if ctx.ifExpr():
            expr = self.get_expr(ctx.ifExpr().expr())
            return f"WHERE {expr}"
        elif ctx.onEventExpression():
            event_desc = self.get_value(ctx.onEventExpression().event_desc())
            return f"ON EVENT {event_desc}"
        elif ctx.everyTimeExpression():
            schedule = self.get_value(ctx.everyTimeExpression().schedule())
            return f"EVERY {schedule}"
        else:
            return ""

    def get_action(self, ctx):
        if ctx.action_value():
            verb = ctx.action_value().action_verb().getText()
            obj = self.get_value(ctx.action_value().action_object())
            return f"EXEC {verb}({obj})"
        else:
            return ""