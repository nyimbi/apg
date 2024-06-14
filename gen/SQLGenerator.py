from DBML4Visitor import DBML4Visitor

class SQLGenerator(DBML4Visitor):
    def __init__(self):
        self.table_name = None
        self.columns = []

    def visitTable(self, ctx):
        self.table_name = ctx.ID().getText()
        table_options = ""
        if ctx.STRING() is not None:
            table_options = f"COMMENT = '{ctx.STRING().getText()[1:-1]}'"

        if ctx.mixin():
            for mixin_ctx in ctx.mixin():
                mixin_name = mixin_ctx.ID().getText()
                self.columns.extend(self.visit(mixin_ctx))

        sql = f"CREATE TABLE {self.table_name} (\n"
        sql += ",\n".join(self.visit(column_ctx) for column_ctx in ctx.column())
        sql += f"\n) {table_options};\n"
        return sql

    def visitColumn(self, ctx):
        name = ctx.ID().getText()
        col_type = self.visit(ctx.type_)
        options = self.visit(ctx.property(0)) if ctx.property() else ""

        return f"{name} {col_type} {options}"

    def visitType(self,ctx):
        col_type = ctx.getText().lower()
        if col_type == "serial":
            return "INTEGER PRIMARY KEY AUTOINCREMENT"
        elif col_type == "float":
            return "REAL"
        elif col_type in ["blob", "file"]:
            return "BLOB"
        elif col_type in ["text"]:
            return "TEXT"
        elif col_type in ["point", "image"]:
            return "BLOB"
        else:
            return col_type.upper()

    def visitProperty(self, ctx):
        if ctx.getText() == "pk":
            return "PRIMARY KEY"
        elif ctx.getText() == "default":
            return f"DEFAULT {self.visit(ctx.STRING())}"
        if ctx.getText() == "required":
            return "NOT NULL"
        elif ctx.getText() == "note":
            return f"COMMENT '{self.visit(ctx.STRING())[1:-1]}'"
        elif ctx.ref():
            target_table = ctx.ref().ID(0).getText()
            target_col = ".".join(ref_ctx.ID().getText() for ref_ctx in ctx.ref().ID()[1:])
            return f"REFERENCES {target_table}({target_col})"
        elif ctx.display():
            display_option = self.visit(ctx.display())
            if display_option.startswith("widget"):
                return f"{display_option.upper()}"
            else:
                return ""
        elif ctx.min():
            return f"CHECK({ctx.ID().getText()} >= {ctx.INT().getText()})"
        elif ctx.max():
            return f"CHECK({ctx.ID().getText()} <= {ctx.INT().getText()})"
        else:
            return ""

    def visitMixin(self, ctx):
        return [self.visit(column_ctx) for column_ctx in ctx.column()]

    def visitView(self, ctx):
        # SQL view statement
        pass

    def visitEnum(self, ctx):
        # SQL enum statement
        pass

    def visitConfig(self, ctx):
        # SQL configuration statement
        pass

    def visitDirective(self, ctx):
        # Directive statement
        pass

    def generate_sql(self, tree):
        return self.visit(tree)