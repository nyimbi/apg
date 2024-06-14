# Generated from /Volumes/Media/src/pjs/appgen/lang/apg.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .apgParser import apgParser
else:
    from apgParser import apgParser

# This class defines a complete generic visitor for a parse tree produced by apgParser.

class apgVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by apgParser#apg.
    def visitApg(self, ctx:apgParser.ApgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#unique.
    def visitUnique(self, ctx:apgParser.UniqueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db.
    def visitDb(self, ctx:apgParser.DbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#int.
    def visitInt(self, ctx:apgParser.IntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#string.
    def visitString(self, ctx:apgParser.StringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ident.
    def visitIdent(self, ctx:apgParser.IdentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#name_attr.
    def visitName_attr(self, ctx:apgParser.Name_attrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#int_list.
    def visitInt_list(self, ctx:apgParser.Int_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ident_list.
    def visitIdent_list(self, ctx:apgParser.Ident_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#string_list.
    def visitString_list(self, ctx:apgParser.String_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#option.
    def visitOption(self, ctx:apgParser.OptionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#option_list.
    def visitOption_list(self, ctx:apgParser.Option_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#importDeclaration.
    def visitImportDeclaration(self, ctx:apgParser.ImportDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#import_file_list.
    def visitImport_file_list(self, ctx:apgParser.Import_file_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#projectBlock.
    def visitProjectBlock(self, ctx:apgParser.ProjectBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#projectName.
    def visitProjectName(self, ctx:apgParser.ProjectNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#project_property_list.
    def visitProject_property_list(self, ctx:apgParser.Project_property_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#project_property.
    def visitProject_property(self, ctx:apgParser.Project_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cloudCfg.
    def visitCloudCfg(self, ctx:apgParser.CloudCfgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cloud_option_list.
    def visitCloud_option_list(self, ctx:apgParser.Cloud_option_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#authCfg.
    def visitAuthCfg(self, ctx:apgParser.AuthCfgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#thirdPartyCfg.
    def visitThirdPartyCfg(self, ctx:apgParser.ThirdPartyCfgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#perfCfg.
    def visitPerfCfg(self, ctx:apgParser.PerfCfgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#versionCfg.
    def visitVersionCfg(self, ctx:apgParser.VersionCfgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#pluginCfg.
    def visitPluginCfg(self, ctx:apgParser.PluginCfgContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#genOptions.
    def visitGenOptions(self, ctx:apgParser.GenOptionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#appGenTarget.
    def visitAppGenTarget(self, ctx:apgParser.AppGenTargetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#language.
    def visitLanguage(self, ctx:apgParser.LanguageContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#lang_list.
    def visitLang_list(self, ctx:apgParser.Lang_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme.
    def visitTheme(self, ctx:apgParser.ThemeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statement.
    def visitStatement(self, ctx:apgParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#object.
    def visitObject(self, ctx:apgParser.ObjectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database.
    def visitDatabase(self, ctx:apgParser.DatabaseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dbname.
    def visitDbname(self, ctx:apgParser.DbnameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_options.
    def visitDatabase_options(self, ctx:apgParser.Database_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mixin.
    def visitMixin(self, ctx:apgParser.MixinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mixin_name.
    def visitMixin_name(self, ctx:apgParser.Mixin_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table.
    def visitTable(self, ctx:apgParser.TableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mixin_list.
    def visitMixin_list(self, ctx:apgParser.Mixin_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_name.
    def visitTable_name(self, ctx:apgParser.Table_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_list.
    def visitColumn_list(self, ctx:apgParser.Column_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column.
    def visitColumn(self, ctx:apgParser.ColumnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_name.
    def visitColumn_name(self, ctx:apgParser.Column_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_type.
    def visitData_type(self, ctx:apgParser.Data_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#varchar.
    def visitVarchar(self, ctx:apgParser.VarcharContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_option_list.
    def visitColumn_option_list(self, ctx:apgParser.Column_option_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_option.
    def visitColumn_option(self, ctx:apgParser.Column_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#primary_key.
    def visitPrimary_key(self, ctx:apgParser.Primary_keyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_default.
    def visitColumn_default(self, ctx:apgParser.Column_defaultContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ref_internal.
    def visitRef_internal(self, ctx:apgParser.Ref_internalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ref_ext.
    def visitRef_ext(self, ctx:apgParser.Ref_extContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ref_name.
    def visitRef_name(self, ctx:apgParser.Ref_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ref_type.
    def visitRef_type(self, ctx:apgParser.Ref_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#oneToOne.
    def visitOneToOne(self, ctx:apgParser.OneToOneContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#oneToMany.
    def visitOneToMany(self, ctx:apgParser.OneToManyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#manyToOne.
    def visitManyToOne(self, ctx:apgParser.ManyToOneContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#manyToMany.
    def visitManyToMany(self, ctx:apgParser.ManyToManyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_name.
    def visitEnum_name(self, ctx:apgParser.Enum_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_internal.
    def visitEnum_internal(self, ctx:apgParser.Enum_internalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_ext.
    def visitEnum_ext(self, ctx:apgParser.Enum_extContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_list.
    def visitEnum_list(self, ctx:apgParser.Enum_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_item.
    def visitEnum_item(self, ctx:apgParser.Enum_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_idx.
    def visitEnum_idx(self, ctx:apgParser.Enum_idxContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_value.
    def visitEnum_value(self, ctx:apgParser.Enum_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#check.
    def visitCheck(self, ctx:apgParser.CheckContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#check_expr.
    def visitCheck_expr(self, ctx:apgParser.Check_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout.
    def visitLayout(self, ctx:apgParser.LayoutContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#note_option.
    def visitNote_option(self, ctx:apgParser.Note_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#note_value.
    def visitNote_value(self, ctx:apgParser.Note_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dbview.
    def visitDbview(self, ctx:apgParser.DbviewContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#view_name.
    def visitView_name(self, ctx:apgParser.View_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_ext.
    def visitIndex_ext(self, ctx:apgParser.Index_extContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_int.
    def visitIndex_int(self, ctx:apgParser.Index_intContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_item_list.
    def visitIndex_item_list(self, ctx:apgParser.Index_item_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_item.
    def visitIndex_item(self, ctx:apgParser.Index_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_names.
    def visitColumn_names(self, ctx:apgParser.Column_namesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_name.
    def visitIndex_name(self, ctx:apgParser.Index_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dbfunc.
    def visitDbfunc(self, ctx:apgParser.DbfuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#func_name.
    def visitFunc_name(self, ctx:apgParser.Func_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#func_body.
    def visitFunc_body(self, ctx:apgParser.Func_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schedule.
    def visitSchedule(self, ctx:apgParser.ScheduleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#field.
    def visitField(self, ctx:apgParser.FieldContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sched_range.
    def visitSched_range(self, ctx:apgParser.Sched_rangeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sched_list.
    def visitSched_list(self, ctx:apgParser.Sched_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#business_rule.
    def visitBusiness_rule(self, ctx:apgParser.Business_ruleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#businessRule.
    def visitBusinessRule(self, ctx:apgParser.BusinessRuleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#IfExpression.
    def visitIfExpression(self, ctx:apgParser.IfExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#AtTimeExpression.
    def visitAtTimeExpression(self, ctx:apgParser.AtTimeExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#EveryTimeExpression.
    def visitEveryTimeExpression(self, ctx:apgParser.EveryTimeExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#OnEventExpression.
    def visitOnEventExpression(self, ctx:apgParser.OnEventExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ifExpr.
    def visitIfExpr(self, ctx:apgParser.IfExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_name.
    def visitRule_name(self, ctx:apgParser.Rule_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_desc.
    def visitEvent_desc(self, ctx:apgParser.Event_descContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#actionExpr.
    def visitActionExpr(self, ctx:apgParser.ActionExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#action_value.
    def visitAction_value(self, ctx:apgParser.Action_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#action_verb.
    def visitAction_verb(self, ctx:apgParser.Action_verbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#action_object.
    def visitAction_object(self, ctx:apgParser.Action_objectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inverseTrigonometricSin.
    def visitInverseTrigonometricSin(self, ctx:apgParser.InverseTrigonometricSinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statisticalMinimum.
    def visitStatisticalMinimum(self, ctx:apgParser.StatisticalMinimumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#binaryAdditionSubtraction.
    def visitBinaryAdditionSubtraction(self, ctx:apgParser.BinaryAdditionSubtractionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#booleanCombination.
    def visitBooleanCombination(self, ctx:apgParser.BooleanCombinationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#nestedExpr.
    def visitNestedExpr(self, ctx:apgParser.NestedExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigonometricCos.
    def visitTrigonometricCos(self, ctx:apgParser.TrigonometricCosContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inverseTrigonometricTan.
    def visitInverseTrigonometricTan(self, ctx:apgParser.InverseTrigonometricTanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#notExpression.
    def visitNotExpression(self, ctx:apgParser.NotExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inverseHyperbolicTangent.
    def visitInverseHyperbolicTangent(self, ctx:apgParser.InverseHyperbolicTangentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#binaryComparison.
    def visitBinaryComparison(self, ctx:apgParser.BinaryComparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigonometricTan.
    def visitTrigonometricTan(self, ctx:apgParser.TrigonometricTanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#literalExpr.
    def visitLiteralExpr(self, ctx:apgParser.LiteralExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#functionCallExpr.
    def visitFunctionCallExpr(self, ctx:apgParser.FunctionCallExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statisticalAverage.
    def visitStatisticalAverage(self, ctx:apgParser.StatisticalAverageContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#binaryMultiplicationDiv.
    def visitBinaryMultiplicationDiv(self, ctx:apgParser.BinaryMultiplicationDivContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#hyperbolicCosine.
    def visitHyperbolicCosine(self, ctx:apgParser.HyperbolicCosineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statisticalMaximum.
    def visitStatisticalMaximum(self, ctx:apgParser.StatisticalMaximumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigonometricSin.
    def visitTrigonometricSin(self, ctx:apgParser.TrigonometricSinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inverseHyperbolicCosine.
    def visitInverseHyperbolicCosine(self, ctx:apgParser.InverseHyperbolicCosineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inverseHyperbolicSine.
    def visitInverseHyperbolicSine(self, ctx:apgParser.InverseHyperbolicSineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statisticalSum.
    def visitStatisticalSum(self, ctx:apgParser.StatisticalSumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#hyperbolicTangent.
    def visitHyperbolicTangent(self, ctx:apgParser.HyperbolicTangentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inverseTrigonometricCos.
    def visitInverseTrigonometricCos(self, ctx:apgParser.InverseTrigonometricCosContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#hyperbolicSine.
    def visitHyperbolicSine(self, ctx:apgParser.HyperbolicSineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#unaryMinus.
    def visitUnaryMinus(self, ctx:apgParser.UnaryMinusContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#identExpression.
    def visitIdentExpression(self, ctx:apgParser.IdentExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#identFunction.
    def visitIdentFunction(self, ctx:apgParser.IdentFunctionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#expr_list.
    def visitExpr_list(self, ctx:apgParser.Expr_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#literal.
    def visitLiteral(self, ctx:apgParser.LiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#booleanOp.
    def visitBooleanOp(self, ctx:apgParser.BooleanOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#comparisonOp.
    def visitComparisonOp(self, ctx:apgParser.ComparisonOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#functionCall.
    def visitFunctionCall(self, ctx:apgParser.FunctionCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#function_name.
    def visitFunction_name(self, ctx:apgParser.Function_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#param_list.
    def visitParam_list(self, ctx:apgParser.Param_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger.
    def visitTrigger(self, ctx:apgParser.TriggerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_name.
    def visitTrigger_name(self, ctx:apgParser.Trigger_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_event.
    def visitTrigger_event(self, ctx:apgParser.Trigger_eventContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_statement.
    def visitTrigger_statement(self, ctx:apgParser.Trigger_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sql_stmt_list.
    def visitSql_stmt_list(self, ctx:apgParser.Sql_stmt_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sql_stmt.
    def visitSql_stmt(self, ctx:apgParser.Sql_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_assignment.
    def visitColumn_assignment(self, ctx:apgParser.Column_assignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#script.
    def visitScript(self, ctx:apgParser.ScriptContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#script_lang.
    def visitScript_lang(self, ctx:apgParser.Script_langContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#script_body.
    def visitScript_body(self, ctx:apgParser.Script_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#script_name.
    def visitScript_name(self, ctx:apgParser.Script_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#workflow.
    def visitWorkflow(self, ctx:apgParser.WorkflowContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#workflow_name.
    def visitWorkflow_name(self, ctx:apgParser.Workflow_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#workflow_step.
    def visitWorkflow_step(self, ctx:apgParser.Workflow_stepContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#step_name.
    def visitStep_name(self, ctx:apgParser.Step_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#workflow_statement.
    def visitWorkflow_statement(self, ctx:apgParser.Workflow_statementContext):
        return self.visitChildren(ctx)



del apgParser