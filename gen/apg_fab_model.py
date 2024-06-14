from apgVisitor import apgVisitor

class ModelGenerator(apgVisitor):

    def visitDatabase(self, ctx):
        db_name = ctx.dbname().getText()
        db_options = self.get_db_options(ctx.database_options())
        return f"""
        class {db_name}(Model):
            {db_options}

            def __unicode__(self):
                return '{db_name}'
        """

    def get_db_options(self, ctx):
        options = [self.get_option(option) for option in ctx.option()]
        return "\n\t\t".join(options)

    def get_option(self, ctx):
        identifier = ctx.ident().getText()
        value = self.get_value(ctx.getChild(2))
        return f"{identifier} = {value}"

    def get_value(self, ctx):
        if ctx.STRING():
            return ctx.STRING().getText()
        elif ctx.INT():
            return int(ctx.INT().getText())
        elif ctx.BOOL():
            return bool(ctx.BOOL().getText())
        elif ctx.FLOAT():
            return float(ctx.FLOAT().getText())
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
        return f"relationship('{table_name}', backref='{ref_name}')"

    def get_enum_internal(self, ctx):
        enum_list = [enum_item.enum_value().getText() for enum_item in ctx.enum_list().enum_item()]
        return f"Enum{enum_list}"

    def get_ref_type(self, ctx):
        if ctx.oneToOne():
            return "uselist=False"
        elif ctx.oneToMany():
            return "lazy='dynamic'"
        elif ctx.manyToOne():
            return "lazy='dynamic'"
        elif ctx.manyToMany():
            return "secondary=association_table"
        else:
            return ""

    def get_table_name(self, ctx):
        return ctx.getText()

    def visitTable(self, ctx):
        table_name = self.get_table_name(ctx.table_name())
        columns = self.get_column_list(ctx.column_list())
        return f"""
        class {table_name}(Model):
            {columns}

            def __unicode__(self):
                return '{table_name}'
        """

    def get_column_list(self, ctx):
        return "\n\t\t".join([self.get_column(column) for column in ctx.column()])

    def get_column(self, ctx):
        col_name = ctx.column_name().getText()
        data_type = self.get_data_type(ctx.data_type())
        options = self.get_column_options(ctx.column_option_list()) if ctx.column_option_list() else ""
        return f"{col_name} = Column({data_type}, {options})"

    def get_data_type(self, ctx):
        if ctx.enum_name():
            enum_name = ctx.enum_name().getText()
            return f"Enum('{enum_name}')"
        elif ctx.BLOB():
            return "LargeBinary"
        elif ctx.STRING():
            return "String"
        elif ctx.BOOL():
            return "Boolean"
        elif ctx.FLOAT():
            return "Float"
        elif ctx.INT():
            return "Integer"
        elif ctx.varchar():
            return f"String({ctx.varchar().getText()})"
        else:
            return ""

    def get_column_options(self, ctx):
        return f"{', '.join([self.get_column_option(option) for option in ctx.column_option()])}"

    def get_column_option(self, ctx):
        if ctx.primary_key():
            return "primary_key=True"
        elif ctx.DEFAULT():
            return self.get_column_default(ctx.column_default())
        elif ctx.NULLABLE():
            return "nullable=True"
        elif ctx.NOT_NULL():
            return "nullable=False"
        elif ctx.unique():
            return "unique=True"
        elif ctx.ref_internal():
            return self.get_ref_internal(ctx.ref_internal())
        elif ctx.enum_internal():
            return self.get_enum_internal(ctx.enum_internal())
        else:
            return ""