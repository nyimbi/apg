from DBML4Visitor import DBML4Visitor

class FlaskAppBuilderViewGenerator(DBML4Visitor):
    def __init__(self):
        self.views = []

    def visitView(self, ctx):
        view_name = ctx.ID().getText()
        model_name = ctx.ref().getText()
        label = view_name.replace('_', ' ').title()
        columns = [c.ID().getText() for c in ctx.column()]

        list_columns = ""
        show_columns = ""
        for col in columns:
            list_columns += f"'{model_name}.{col}',"
            show_columns += f"{col},"

        list_columns += "'actions'"
        show_columns += "get_actions,"

        view = f"""
class {view_name}View(CustomModelView):
    datamodel = SQLAInterface({model_name})
    label = '{label}'
    list_columns = ({list_columns})
    show_columns = ({show_columns})
    edit_columns = ["{', '.join(columns)}"]
    add_columns = [{', '.join(columns)}]
    search_columns = ({list_columns.replace("'actions'", "")})
    add_exclude_columns = ['id']
    edit_exclude_columns = ['id']
    show_exclude_columns = ['id']

    @action('delete', 'Delete', 'Are you sure you want to delete selected records?')
    def delete(self, items):
        for item in items:
            self.datamodel.session.delete(item)
        self.datamodel.session.commit()
        self.update_redirect()

    @action('multiaction', 'Multi Action', 'Do you really want to perform this action?')
    def multiaction(self, items):
        pass
"""

        self.views.append(view)
        return ""

    def generate_flask_appbuilder_views(self, tree):
        for statement_context in tree.statement():
            if statement_context.view():
                self.visit(statement_context.view())

        return "\n\n".join(self.views)