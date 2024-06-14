# Generated from /Volumes/Media/src/pjs/appgen/lang/appgen.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .appgenParser import appgenParser
else:
    from appgenParser import appgenParser

# This class defines a complete generic visitor for a parse tree produced by appgenParser.

class appgenVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by appgenParser#unique.
    def visitUnique(self, ctx:appgenParser.UniqueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#db.
    def visitDb(self, ctx:appgenParser.DbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#int.
    def visitInt(self, ctx:appgenParser.IntContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#string.
    def visitString(self, ctx:appgenParser.StringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ident.
    def visitIdent(self, ctx:appgenParser.IdentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#name_attr.
    def visitName_attr(self, ctx:appgenParser.Name_attrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#int_list.
    def visitInt_list(self, ctx:appgenParser.Int_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ident_list.
    def visitIdent_list(self, ctx:appgenParser.Ident_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#string_list.
    def visitString_list(self, ctx:appgenParser.String_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#option.
    def visitOption(self, ctx:appgenParser.OptionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#option_list.
    def visitOption_list(self, ctx:appgenParser.Option_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#appgen.
    def visitAppgen(self, ctx:appgenParser.AppgenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#importDeclaration.
    def visitImportDeclaration(self, ctx:appgenParser.ImportDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#import_file_list.
    def visitImport_file_list(self, ctx:appgenParser.Import_file_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#import_file_name.
    def visitImport_file_name(self, ctx:appgenParser.Import_file_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#projectBlock.
    def visitProjectBlock(self, ctx:appgenParser.ProjectBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#project_name.
    def visitProject_name(self, ctx:appgenParser.Project_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#project_property_list.
    def visitProject_property_list(self, ctx:appgenParser.Project_property_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#project_property.
    def visitProject_property(self, ctx:appgenParser.Project_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#gen_option.
    def visitGen_option(self, ctx:appgenParser.Gen_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#app_gen_target.
    def visitApp_gen_target(self, ctx:appgenParser.App_gen_targetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#deployment.
    def visitDeployment(self, ctx:appgenParser.DeploymentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#deployment_option_list.
    def visitDeployment_option_list(self, ctx:appgenParser.Deployment_option_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#language.
    def visitLanguage(self, ctx:appgenParser.LanguageContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#lang_list.
    def visitLang_list(self, ctx:appgenParser.Lang_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#theme.
    def visitTheme(self, ctx:appgenParser.ThemeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#report_spec.
    def visitReport_spec(self, ctx:appgenParser.Report_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#report_name.
    def visitReport_name(self, ctx:appgenParser.Report_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#report_property_list.
    def visitReport_property_list(self, ctx:appgenParser.Report_property_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#report_property.
    def visitReport_property(self, ctx:appgenParser.Report_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#chart_specification.
    def visitChart_specification(self, ctx:appgenParser.Chart_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#chart_name.
    def visitChart_name(self, ctx:appgenParser.Chart_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#chart_property_list.
    def visitChart_property_list(self, ctx:appgenParser.Chart_property_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#chart_property.
    def visitChart_property(self, ctx:appgenParser.Chart_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#config.
    def visitConfig(self, ctx:appgenParser.ConfigContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#config_options_list.
    def visitConfig_options_list(self, ctx:appgenParser.Config_options_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#config_option.
    def visitConfig_option(self, ctx:appgenParser.Config_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#statement.
    def visitStatement(self, ctx:appgenParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#dbfunc.
    def visitDbfunc(self, ctx:appgenParser.DbfuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#func_name.
    def visitFunc_name(self, ctx:appgenParser.Func_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#object.
    def visitObject(self, ctx:appgenParser.ObjectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#database.
    def visitDatabase(self, ctx:appgenParser.DatabaseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#schema.
    def visitSchema(self, ctx:appgenParser.SchemaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#mixin.
    def visitMixin(self, ctx:appgenParser.MixinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#mixin_name.
    def visitMixin_name(self, ctx:appgenParser.Mixin_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_list.
    def visitColumn_list(self, ctx:appgenParser.Column_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column.
    def visitColumn(self, ctx:appgenParser.ColumnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_name.
    def visitColumn_name(self, ctx:appgenParser.Column_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_option_list.
    def visitColumn_option_list(self, ctx:appgenParser.Column_option_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#primary_key.
    def visitPrimary_key(self, ctx:appgenParser.Primary_keyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_option.
    def visitColumn_option(self, ctx:appgenParser.Column_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#check_expr.
    def visitCheck_expr(self, ctx:appgenParser.Check_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#data_type.
    def visitData_type(self, ctx:appgenParser.Data_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_reference.
    def visitColumn_reference(self, ctx:appgenParser.Column_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#table_reference.
    def visitTable_reference(self, ctx:appgenParser.Table_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_default.
    def visitColumn_default(self, ctx:appgenParser.Column_defaultContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_name.
    def visitEnum_name(self, ctx:appgenParser.Enum_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_internal.
    def visitEnum_internal(self, ctx:appgenParser.Enum_internalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_out.
    def visitEnum_out(self, ctx:appgenParser.Enum_outContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_list.
    def visitEnum_list(self, ctx:appgenParser.Enum_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_item.
    def visitEnum_item(self, ctx:appgenParser.Enum_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_idx.
    def visitEnum_idx(self, ctx:appgenParser.Enum_idxContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#enum_value.
    def visitEnum_value(self, ctx:appgenParser.Enum_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#display_method.
    def visitDisplay_method(self, ctx:appgenParser.Display_methodContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#note_option.
    def visitNote_option(self, ctx:appgenParser.Note_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#note_value.
    def visitNote_value(self, ctx:appgenParser.Note_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#varchar.
    def visitVarchar(self, ctx:appgenParser.VarcharContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#tableDecl.
    def visitTableDecl(self, ctx:appgenParser.TableDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#mixin_list.
    def visitMixin_list(self, ctx:appgenParser.Mixin_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#table_name.
    def visitTable_name(self, ctx:appgenParser.Table_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#dbview.
    def visitDbview(self, ctx:appgenParser.DbviewContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#db_join.
    def visitDb_join(self, ctx:appgenParser.Db_joinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ref_internal.
    def visitRef_internal(self, ctx:appgenParser.Ref_internalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ext_ref.
    def visitExt_ref(self, ctx:appgenParser.Ext_refContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ref_name.
    def visitRef_name(self, ctx:appgenParser.Ref_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ref_type.
    def visitRef_type(self, ctx:appgenParser.Ref_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#oneToOne.
    def visitOneToOne(self, ctx:appgenParser.OneToOneContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#oneToMany.
    def visitOneToMany(self, ctx:appgenParser.OneToManyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#manyToOne.
    def visitManyToOne(self, ctx:appgenParser.ManyToOneContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#manyToMany.
    def visitManyToMany(self, ctx:appgenParser.ManyToManyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#index_ext.
    def visitIndex_ext(self, ctx:appgenParser.Index_extContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#index_int.
    def visitIndex_int(self, ctx:appgenParser.Index_intContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#index_item_list.
    def visitIndex_item_list(self, ctx:appgenParser.Index_item_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#index_item.
    def visitIndex_item(self, ctx:appgenParser.Index_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#column_names.
    def visitColumn_names(self, ctx:appgenParser.Column_namesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#index_name.
    def visitIndex_name(self, ctx:appgenParser.Index_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#view_s_spec.
    def visitView_s_spec(self, ctx:appgenParser.View_s_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#view_spec_list.
    def visitView_spec_list(self, ctx:appgenParser.View_spec_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#view_spec.
    def visitView_spec(self, ctx:appgenParser.View_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#view_type.
    def visitView_type(self, ctx:appgenParser.View_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#view_spec_options.
    def visitView_spec_options(self, ctx:appgenParser.View_spec_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#business_rule.
    def visitBusiness_rule(self, ctx:appgenParser.Business_ruleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#businessRule.
    def visitBusinessRule(self, ctx:appgenParser.BusinessRuleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#ifExpr.
    def visitIfExpr(self, ctx:appgenParser.IfExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#rule_name.
    def visitRule_name(self, ctx:appgenParser.Rule_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#actionExpr.
    def visitActionExpr(self, ctx:appgenParser.ActionExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#python_code.
    def visitPython_code(self, ctx:appgenParser.Python_codeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#sms.
    def visitSms(self, ctx:appgenParser.SmsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#notify.
    def visitNotify(self, ctx:appgenParser.NotifyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#search.
    def visitSearch(self, ctx:appgenParser.SearchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#flag.
    def visitFlag(self, ctx:appgenParser.FlagContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#upload.
    def visitUpload(self, ctx:appgenParser.UploadContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#download.
    def visitDownload(self, ctx:appgenParser.DownloadContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#execute_query.
    def visitExecute_query(self, ctx:appgenParser.Execute_queryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#destination.
    def visitDestination(self, ctx:appgenParser.DestinationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#server_loc.
    def visitServer_loc(self, ctx:appgenParser.Server_locContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#inverseTrigonometricSin.
    def visitInverseTrigonometricSin(self, ctx:appgenParser.InverseTrigonometricSinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#statisticalMinimum.
    def visitStatisticalMinimum(self, ctx:appgenParser.StatisticalMinimumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#binaryAdditionSubtraction.
    def visitBinaryAdditionSubtraction(self, ctx:appgenParser.BinaryAdditionSubtractionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#booleanCombination.
    def visitBooleanCombination(self, ctx:appgenParser.BooleanCombinationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#nestedExpr.
    def visitNestedExpr(self, ctx:appgenParser.NestedExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#trigonometricCos.
    def visitTrigonometricCos(self, ctx:appgenParser.TrigonometricCosContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#inverseTrigonometricTan.
    def visitInverseTrigonometricTan(self, ctx:appgenParser.InverseTrigonometricTanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#inverseHyperbolicTangent.
    def visitInverseHyperbolicTangent(self, ctx:appgenParser.InverseHyperbolicTangentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#binaryComparison.
    def visitBinaryComparison(self, ctx:appgenParser.BinaryComparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#trigonometricTan.
    def visitTrigonometricTan(self, ctx:appgenParser.TrigonometricTanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#literalExpr.
    def visitLiteralExpr(self, ctx:appgenParser.LiteralExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#functionCallExpr.
    def visitFunctionCallExpr(self, ctx:appgenParser.FunctionCallExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#statisticalAverage.
    def visitStatisticalAverage(self, ctx:appgenParser.StatisticalAverageContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#binaryMultiplicationDiv.
    def visitBinaryMultiplicationDiv(self, ctx:appgenParser.BinaryMultiplicationDivContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#hyperbolicCosine.
    def visitHyperbolicCosine(self, ctx:appgenParser.HyperbolicCosineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#statisticalMaximum.
    def visitStatisticalMaximum(self, ctx:appgenParser.StatisticalMaximumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#trigonometricSin.
    def visitTrigonometricSin(self, ctx:appgenParser.TrigonometricSinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#inverseHyperbolicCosine.
    def visitInverseHyperbolicCosine(self, ctx:appgenParser.InverseHyperbolicCosineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#inverseHyperbolicSine.
    def visitInverseHyperbolicSine(self, ctx:appgenParser.InverseHyperbolicSineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#statisticalSum.
    def visitStatisticalSum(self, ctx:appgenParser.StatisticalSumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#hyperbolicTangent.
    def visitHyperbolicTangent(self, ctx:appgenParser.HyperbolicTangentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#inverseTrigonometricCos.
    def visitInverseTrigonometricCos(self, ctx:appgenParser.InverseTrigonometricCosContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#hyperbolicSine.
    def visitHyperbolicSine(self, ctx:appgenParser.HyperbolicSineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#unaryMinus.
    def visitUnaryMinus(self, ctx:appgenParser.UnaryMinusContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#identExpression.
    def visitIdentExpression(self, ctx:appgenParser.IdentExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#expr_list.
    def visitExpr_list(self, ctx:appgenParser.Expr_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#literal.
    def visitLiteral(self, ctx:appgenParser.LiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#booleanOp.
    def visitBooleanOp(self, ctx:appgenParser.BooleanOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#comparisonOp.
    def visitComparisonOp(self, ctx:appgenParser.ComparisonOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#arithmeticOp.
    def visitArithmeticOp(self, ctx:appgenParser.ArithmeticOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#functionCall.
    def visitFunctionCall(self, ctx:appgenParser.FunctionCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#function_name.
    def visitFunction_name(self, ctx:appgenParser.Function_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by appgenParser#param_list.
    def visitParam_list(self, ctx:appgenParser.Param_listContext):
        return self.visitChildren(ctx)



del appgenParser