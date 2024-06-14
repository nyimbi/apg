from appgenVisitor import appgenVisitor

class FlaskAppBuilderModelGenerator(DBML4Visitor):
    def __init__(self):
        self.table_name = None
        self.columns = []

    def visitTable(self, ctx):
        table_name = ctx.ID().getText()
        columns = []
        relationships = []
        for column_ctx in ctx.column():
            col_name = column_ctx.ID().getText()
            col_type = column_ctx.type().getText().upper()
            columns.append((col_name, col_type))

        for ref_ctx in ctx.reference():
            if ref_ctx.many_to_many():
                ref_table_name = ref_ctx.table_name().getText()
                association_table_name = f"{table_name}_{ref_table_name}"
                relationships.append((ref_table_name, association_table_name))

        model = f"""
    class {table_name}(Base):
        __tablename__ = '{table_name.lower()}'

    """
        for col_name, col_type in columns:
            model += f"    {col_name} = Column({col_type})\n"

        for ref_table_name, association_table_name in relationships:
            model += f"""
    {association_table_name} = Table('{association_table_name.lower()}', Base.metadata,
        Column('{table_name.lower()}_id', Integer, ForeignKey('{table_name.lower()}.id')),
        Column('{ref_table_name.lower()}_id', Integer, ForeignKey('{ref_table_name.lower()}.id'))
    )
    """

        model += f"""
        def __repr__(self):
            return "<{table_name} {self.id}>"
    """

        self.models.append(model)
        return ""

    def visitMixin(self, ctx):
        return [self.visit(column_ctx) for column_ctx in ctx.column()]

    def visitColumn(self, ctx):
        name = ctx.ID().getText()
        col_type = self.visit(ctx.type_)
        options = [self.visit(o) for o in ctx.property()]

        if "pk" in options:
            col_type = "Integer(primary_key=True)"
            options.remove("pk")
        elif "?" in col_type:
            col_type = col_type.replace("?", "")
            options.append("nullable=True")

        sqla_type = f"{col_type[0].upper()}{col_type[1:]}"
        definition = f"Column({sqla_type}{', '.join(options)}, unique=False, nullable=False)"

        return f"{name} = {definition}"

    def visitProperty(self, ctx):
        if ctx.getText() == "pk":
            return ""
        elif ctx.getText() == "default":
            return f"default={self.visit(ctx.STRING())}"
        elif ctx.getText() == "required":
            return ""
        elif ctx.getText() == "note":
            return f"comment='{self.visit(ctx.STRING())[1:-1]}'"
        elif ctx.ref():
            target_table = ctx.ref().ID(0).getText()
            target_col = ".".join(ref_ctx.ID().getText() for ref_ctx in ctx.ref().ID()[1:])
            return f"ForeignKey('{target_table}.{target_col}')"
        elif ctx.display():
            display_option = self.visit(ctx.display())
            if display_option.startswith("widget"):
                return f""
            elif display_option.startswith("hide"):
                return f""
            elif display_option.startswith("hint"):
                return f""
            elif display_option.startswith("display"):
                return f"info='{display_option.split('=')[1]}'"
            elif display_option.startswith("tab"):
                return f""
            elif display_option.startswith("sequence"):
                return f"order={display_option.split('=')[1]}"
        elif ctx.min():
            return f""
        elif ctx.max():
            return f""
        else:
            return ""

    def generate_flask_appbuilder_models(self, tree):
        models = ""
        for statement_context in tree.statement():
            if statement_context.table():
                models += self.visit(statement_context.table()) + "\n\n"
        return models