from apgListener import apgListener

class ViewGenerator(apgListener):

    def __init__(self, output_file):
        self.output_file = output_file

    def enterTable(self, ctx):
        table_name = ctx.table_name().getText()
        self.output_file.write(f"""
class {table_name}ModelView(ModelView):
    datamodel = SQLAInterface({table_name})
    add_columns = [")
    """)

    def exitColumn(self, ctx):
        col_name = ctx.column_name().getText()
        self.output_file.write(f"'{col_name}', ")

    def exitTable(self, ctx):
        self.output_file.write(f"""[
    edit_columns = add_columns
    list_columns = add_columns
    show_columns = add_columns
    page_size = 50
    """)  # adding a newline at the end of each view definition