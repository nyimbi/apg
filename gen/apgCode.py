from apgParser import apgParser
from apgVisitor import apgVisitor
from apgListener import apgListener

class GenerateSqlListener(apgParser):
    def __init__(self):
        self._tables = []
        self._sql = ''

    def get_sql(self):
        return self._sql

    def exitSchema(self, ctx: apgParser.SchemaContext):
        self._sql += f"CREATE SCHEMA {ctx.schema_name().getText()};\n"

    def exitTable_def(self, ctx: apgParser.Table_defContext):
        table_sql = f"CREATE TABLE {ctx.table_name().getText()} (\n"
        for column in ctx.column_list().column():
            table_sql += self.get_column_sql(column) + ',\n'
        for constraint in ctx.constraint():
            table_sql += self.get_constraint_sql(constraint) + '\n'
        table_sql += ");"
        self._tables.append(ctx.table_name().getText())
        self._sql += table_sql + '\n'

    def exitMixin(self, ctx: apgParser.MixinContext):
        mixin_sql = ''
        columns = []  # store columns object
        for col in ctx.column_list().column():
            columns.append(self.get_column_sql(col))
        if columns:
            mixin_sql = f"ALTER TABLE {ctx.mixin_name().getText()} ADD COLUMN {', '.join(columns)};"
        self._sql += mixin_sql + '\n'

    def get_column_sql(self, column):
        column_name = column.column_name().getText()
        data_type = column.data_type().getText()
        nullable = ''
        if column.NULLABLE():
            nullable = 'NULL'
        else:
            nullable = 'NOT NULL'
        return f"{column_name} {data_type} {nullable}"

    def get_constraint_sql(self, constraint):
        if constraint.primary_key():
            return self.get_primary_key_sql(constraint.primary_key())
        elif constraint.foreign_key():
            return self.get_foreign_key_sql(constraint.foreign_key())

    def get_primary_key_sql(self, constraint):
        pk_columns = []  # store column names
        for col in constraint.column_list().column():
            pk_columns.append(col.column_name().getText())
        return f"PRIMARY KEY ({','.join(pk_columns)})"

    def get_foreign_key_sql(self, constraint):
        fk_columns = []  # store column names
        ref_table = constraint.table_ref().getText()
        ref_columns = []  # store column names in the referenced table
        for col in constraint.column_list().column():
            fk_columns.append(col.column_name().getText())
            ref_columns.append(col.ref_column_name().getText())
        fk_sql = f"FOREIGN KEY ({','.join(fk_columns)}) REFERENCES {ref_table}({','.join(ref_columns)}) MATCH SIMPLE"
        if constraint.on_delete():
            fk_sql += f" {self.get_on_delete_sql(constraint.on_delete())}"
        if constraint.on_update():
            fk_sql += f" {self.get_on_update_sql(constraint.on_update())}"
        return fk_sql

    def get_on_delete_sql(self, on_delete):
        if on_delete.CASCADE():
            return 'ON DELETE CASCADE'
        elif on_delete.SET_NULL():
            return 'ON DELETE SET NULL'
        else:
            return ''

    def get_on_update_sql(self, on_update):
        if on_update.CASCADE():
            return 'ON UPDATE CASCADE'
        elif on_update.SET_NULL():
            return 'ON UPDATE SET NULL'
        else:
            return ''



class GenerateSqlVisitor(apgVisitor):
    def __init__(self):
        self._tables = []
        self._sql = ''

    def get_sql(self):
        return self._sql

    def visitSchema(self, ctx:apgParser.SchemaContext):
        self._sql += f"CREATE SCHEMA {ctx.schema_name().getText()};\n"

    def visitTable_def(self, ctx:apgParser.Table_defContext):
        table_sql = f"CREATE TABLE {ctx.table_name().getText()} (\n"
        table_sql += self.visitColumn_list(ctx.column_list())
        table_sql += "\n);"

        self._tables.append(ctx.table_name().getText())
        self._sql += table_sql + '\n'

    def visitMixin(self, ctx:apgParser.MixinContext):
        mixin_sql = ''
        columns = self.visitColumn_list(ctx.column_list())
        if columns:
            mixin_sql = f"ALTER TABLE {ctx.mixin_name().getText()} ADD COLUMN {columns.strip()};\n"
        return mixin_sql

    def visitColumn_list(self, ctx:apgParser.Column_listContext):
        column_sql = ''
        columns = []
        for column in ctx.column():
            columns.append(column)
        for i in range(len(columns)):
            column = columns[i]
            temp = ' ' + self.visit(column)
            if i != len(columns) - 1:
                temp = temp + ','
            column_sql += temp + '\n'
        return column_sql.strip(',')

    def visitColumn(self, ctx:apgParser.ColumnContext):
        column_name = self.visit(ctx.column_name())
        data_type = self.visit(ctx.data_type())
        nullable = ''
        if ctx.NULLABLE():
            nullable = 'NULL'
        else:
            nullable = 'NOT NULL'
        return f"{column_name} {data_type} {nullable}"

    def visitPrimary_key(self, ctx:apgParser.Primary_keyContext):
        pk_columns = self.visitColumn_list(ctx.column_list())
        pk_sql = f"ALTER TABLE {ctx.table_name().getText()} ADD PRIMARY KEY ({pk_columns.strip()});"
        self._sql += pk_sql + '\n'

    def visitForeign_key(self, ctx:apgParser.Foreign_keyContext):
        fk_columns = self.visitColumn_list(ctx.column_list())
        ref_table = ctx.table_name().getText()
        ref_columns = self.visitColumn_list(ctx.ref_column_list())

        fk_sql = f"ALTER TABLE {ctx.mixin_name().getText()} ADD CONSTRAINT {ctx.constraint_name().getText()} "
        fk_sql += f"FOREIGN KEY ({fk_columns.strip()}) REFERENCES {ref_table}({ref_columns.strip()}) MATCH SIMPLE "
        if ctx.on_delete():
            fk_sql += self.visitOn_delete(ctx.on_delete())
        else:
            fk_sql += "ON DELETE NO ACTION"
        if ctx.on_update():
            fk_sql += self.visitOn_update(ctx.on_update())
        else:
            fk_sql += " ON UPDATE NO ACTION"
        self._sql += fk_sql + '\n'

    def visitOn_delete(self, ctx:apgParser.On_deleteContext):
        if ctx.CASCADE():
            return 'ON DELETE CASCADE'
        elif ctx.SET_NULL():
            return 'ON DELETE SET NULL'
        else:
            return ''

    def visitOn_update(self, ctx:apgParser.On_updateContext):
        if ctx.CASCADE():
            return 'ON UPDATE CASCADE'
        elif ctx.SET_NULL():
            return 'ON UPDATE SET NULL'
        else:
            return ''

