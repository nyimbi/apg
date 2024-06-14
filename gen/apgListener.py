# Generated from /Volumes/Media/src/pjs/appgen/lang/apg.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .apgParser import apgParser
else:
    from apgParser import apgParser

# This class defines a complete listener for a parse tree produced by apgParser.
class apgListener(ParseTreeListener):

    # Enter a parse tree produced by apgParser#apg.
    def enterApg(self, ctx:apgParser.ApgContext):
        pass

    # Exit a parse tree produced by apgParser#apg.
    def exitApg(self, ctx:apgParser.ApgContext):
        pass


    # Enter a parse tree produced by apgParser#unique.
    def enterUnique(self, ctx:apgParser.UniqueContext):
        pass

    # Exit a parse tree produced by apgParser#unique.
    def exitUnique(self, ctx:apgParser.UniqueContext):
        pass


    # Enter a parse tree produced by apgParser#db.
    def enterDb(self, ctx:apgParser.DbContext):
        pass

    # Exit a parse tree produced by apgParser#db.
    def exitDb(self, ctx:apgParser.DbContext):
        pass


    # Enter a parse tree produced by apgParser#int.
    def enterInt(self, ctx:apgParser.IntContext):
        pass

    # Exit a parse tree produced by apgParser#int.
    def exitInt(self, ctx:apgParser.IntContext):
        pass


    # Enter a parse tree produced by apgParser#string.
    def enterString(self, ctx:apgParser.StringContext):
        pass

    # Exit a parse tree produced by apgParser#string.
    def exitString(self, ctx:apgParser.StringContext):
        pass


    # Enter a parse tree produced by apgParser#ident.
    def enterIdent(self, ctx:apgParser.IdentContext):
        pass

    # Exit a parse tree produced by apgParser#ident.
    def exitIdent(self, ctx:apgParser.IdentContext):
        pass


    # Enter a parse tree produced by apgParser#name_attr.
    def enterName_attr(self, ctx:apgParser.Name_attrContext):
        pass

    # Exit a parse tree produced by apgParser#name_attr.
    def exitName_attr(self, ctx:apgParser.Name_attrContext):
        pass


    # Enter a parse tree produced by apgParser#int_list.
    def enterInt_list(self, ctx:apgParser.Int_listContext):
        pass

    # Exit a parse tree produced by apgParser#int_list.
    def exitInt_list(self, ctx:apgParser.Int_listContext):
        pass


    # Enter a parse tree produced by apgParser#ident_list.
    def enterIdent_list(self, ctx:apgParser.Ident_listContext):
        pass

    # Exit a parse tree produced by apgParser#ident_list.
    def exitIdent_list(self, ctx:apgParser.Ident_listContext):
        pass


    # Enter a parse tree produced by apgParser#string_list.
    def enterString_list(self, ctx:apgParser.String_listContext):
        pass

    # Exit a parse tree produced by apgParser#string_list.
    def exitString_list(self, ctx:apgParser.String_listContext):
        pass


    # Enter a parse tree produced by apgParser#option.
    def enterOption(self, ctx:apgParser.OptionContext):
        pass

    # Exit a parse tree produced by apgParser#option.
    def exitOption(self, ctx:apgParser.OptionContext):
        pass


    # Enter a parse tree produced by apgParser#option_list.
    def enterOption_list(self, ctx:apgParser.Option_listContext):
        pass

    # Exit a parse tree produced by apgParser#option_list.
    def exitOption_list(self, ctx:apgParser.Option_listContext):
        pass


    # Enter a parse tree produced by apgParser#importDeclaration.
    def enterImportDeclaration(self, ctx:apgParser.ImportDeclarationContext):
        pass

    # Exit a parse tree produced by apgParser#importDeclaration.
    def exitImportDeclaration(self, ctx:apgParser.ImportDeclarationContext):
        pass


    # Enter a parse tree produced by apgParser#import_file_list.
    def enterImport_file_list(self, ctx:apgParser.Import_file_listContext):
        pass

    # Exit a parse tree produced by apgParser#import_file_list.
    def exitImport_file_list(self, ctx:apgParser.Import_file_listContext):
        pass


    # Enter a parse tree produced by apgParser#projectBlock.
    def enterProjectBlock(self, ctx:apgParser.ProjectBlockContext):
        pass

    # Exit a parse tree produced by apgParser#projectBlock.
    def exitProjectBlock(self, ctx:apgParser.ProjectBlockContext):
        pass


    # Enter a parse tree produced by apgParser#projectName.
    def enterProjectName(self, ctx:apgParser.ProjectNameContext):
        pass

    # Exit a parse tree produced by apgParser#projectName.
    def exitProjectName(self, ctx:apgParser.ProjectNameContext):
        pass


    # Enter a parse tree produced by apgParser#project_property_list.
    def enterProject_property_list(self, ctx:apgParser.Project_property_listContext):
        pass

    # Exit a parse tree produced by apgParser#project_property_list.
    def exitProject_property_list(self, ctx:apgParser.Project_property_listContext):
        pass


    # Enter a parse tree produced by apgParser#project_property.
    def enterProject_property(self, ctx:apgParser.Project_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#project_property.
    def exitProject_property(self, ctx:apgParser.Project_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#cloudCfg.
    def enterCloudCfg(self, ctx:apgParser.CloudCfgContext):
        pass

    # Exit a parse tree produced by apgParser#cloudCfg.
    def exitCloudCfg(self, ctx:apgParser.CloudCfgContext):
        pass


    # Enter a parse tree produced by apgParser#cloud_option_list.
    def enterCloud_option_list(self, ctx:apgParser.Cloud_option_listContext):
        pass

    # Exit a parse tree produced by apgParser#cloud_option_list.
    def exitCloud_option_list(self, ctx:apgParser.Cloud_option_listContext):
        pass


    # Enter a parse tree produced by apgParser#authCfg.
    def enterAuthCfg(self, ctx:apgParser.AuthCfgContext):
        pass

    # Exit a parse tree produced by apgParser#authCfg.
    def exitAuthCfg(self, ctx:apgParser.AuthCfgContext):
        pass


    # Enter a parse tree produced by apgParser#thirdPartyCfg.
    def enterThirdPartyCfg(self, ctx:apgParser.ThirdPartyCfgContext):
        pass

    # Exit a parse tree produced by apgParser#thirdPartyCfg.
    def exitThirdPartyCfg(self, ctx:apgParser.ThirdPartyCfgContext):
        pass


    # Enter a parse tree produced by apgParser#perfCfg.
    def enterPerfCfg(self, ctx:apgParser.PerfCfgContext):
        pass

    # Exit a parse tree produced by apgParser#perfCfg.
    def exitPerfCfg(self, ctx:apgParser.PerfCfgContext):
        pass


    # Enter a parse tree produced by apgParser#versionCfg.
    def enterVersionCfg(self, ctx:apgParser.VersionCfgContext):
        pass

    # Exit a parse tree produced by apgParser#versionCfg.
    def exitVersionCfg(self, ctx:apgParser.VersionCfgContext):
        pass


    # Enter a parse tree produced by apgParser#pluginCfg.
    def enterPluginCfg(self, ctx:apgParser.PluginCfgContext):
        pass

    # Exit a parse tree produced by apgParser#pluginCfg.
    def exitPluginCfg(self, ctx:apgParser.PluginCfgContext):
        pass


    # Enter a parse tree produced by apgParser#genOptions.
    def enterGenOptions(self, ctx:apgParser.GenOptionsContext):
        pass

    # Exit a parse tree produced by apgParser#genOptions.
    def exitGenOptions(self, ctx:apgParser.GenOptionsContext):
        pass


    # Enter a parse tree produced by apgParser#appGenTarget.
    def enterAppGenTarget(self, ctx:apgParser.AppGenTargetContext):
        pass

    # Exit a parse tree produced by apgParser#appGenTarget.
    def exitAppGenTarget(self, ctx:apgParser.AppGenTargetContext):
        pass


    # Enter a parse tree produced by apgParser#language.
    def enterLanguage(self, ctx:apgParser.LanguageContext):
        pass

    # Exit a parse tree produced by apgParser#language.
    def exitLanguage(self, ctx:apgParser.LanguageContext):
        pass


    # Enter a parse tree produced by apgParser#lang_list.
    def enterLang_list(self, ctx:apgParser.Lang_listContext):
        pass

    # Exit a parse tree produced by apgParser#lang_list.
    def exitLang_list(self, ctx:apgParser.Lang_listContext):
        pass


    # Enter a parse tree produced by apgParser#theme.
    def enterTheme(self, ctx:apgParser.ThemeContext):
        pass

    # Exit a parse tree produced by apgParser#theme.
    def exitTheme(self, ctx:apgParser.ThemeContext):
        pass


    # Enter a parse tree produced by apgParser#statement.
    def enterStatement(self, ctx:apgParser.StatementContext):
        pass

    # Exit a parse tree produced by apgParser#statement.
    def exitStatement(self, ctx:apgParser.StatementContext):
        pass


    # Enter a parse tree produced by apgParser#object.
    def enterObject(self, ctx:apgParser.ObjectContext):
        pass

    # Exit a parse tree produced by apgParser#object.
    def exitObject(self, ctx:apgParser.ObjectContext):
        pass


    # Enter a parse tree produced by apgParser#database.
    def enterDatabase(self, ctx:apgParser.DatabaseContext):
        pass

    # Exit a parse tree produced by apgParser#database.
    def exitDatabase(self, ctx:apgParser.DatabaseContext):
        pass


    # Enter a parse tree produced by apgParser#dbname.
    def enterDbname(self, ctx:apgParser.DbnameContext):
        pass

    # Exit a parse tree produced by apgParser#dbname.
    def exitDbname(self, ctx:apgParser.DbnameContext):
        pass


    # Enter a parse tree produced by apgParser#database_options.
    def enterDatabase_options(self, ctx:apgParser.Database_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#database_options.
    def exitDatabase_options(self, ctx:apgParser.Database_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#mixin.
    def enterMixin(self, ctx:apgParser.MixinContext):
        pass

    # Exit a parse tree produced by apgParser#mixin.
    def exitMixin(self, ctx:apgParser.MixinContext):
        pass


    # Enter a parse tree produced by apgParser#mixin_name.
    def enterMixin_name(self, ctx:apgParser.Mixin_nameContext):
        pass

    # Exit a parse tree produced by apgParser#mixin_name.
    def exitMixin_name(self, ctx:apgParser.Mixin_nameContext):
        pass


    # Enter a parse tree produced by apgParser#table.
    def enterTable(self, ctx:apgParser.TableContext):
        pass

    # Exit a parse tree produced by apgParser#table.
    def exitTable(self, ctx:apgParser.TableContext):
        pass


    # Enter a parse tree produced by apgParser#mixin_list.
    def enterMixin_list(self, ctx:apgParser.Mixin_listContext):
        pass

    # Exit a parse tree produced by apgParser#mixin_list.
    def exitMixin_list(self, ctx:apgParser.Mixin_listContext):
        pass


    # Enter a parse tree produced by apgParser#table_name.
    def enterTable_name(self, ctx:apgParser.Table_nameContext):
        pass

    # Exit a parse tree produced by apgParser#table_name.
    def exitTable_name(self, ctx:apgParser.Table_nameContext):
        pass


    # Enter a parse tree produced by apgParser#column_list.
    def enterColumn_list(self, ctx:apgParser.Column_listContext):
        pass

    # Exit a parse tree produced by apgParser#column_list.
    def exitColumn_list(self, ctx:apgParser.Column_listContext):
        pass


    # Enter a parse tree produced by apgParser#column.
    def enterColumn(self, ctx:apgParser.ColumnContext):
        pass

    # Exit a parse tree produced by apgParser#column.
    def exitColumn(self, ctx:apgParser.ColumnContext):
        pass


    # Enter a parse tree produced by apgParser#column_name.
    def enterColumn_name(self, ctx:apgParser.Column_nameContext):
        pass

    # Exit a parse tree produced by apgParser#column_name.
    def exitColumn_name(self, ctx:apgParser.Column_nameContext):
        pass


    # Enter a parse tree produced by apgParser#data_type.
    def enterData_type(self, ctx:apgParser.Data_typeContext):
        pass

    # Exit a parse tree produced by apgParser#data_type.
    def exitData_type(self, ctx:apgParser.Data_typeContext):
        pass


    # Enter a parse tree produced by apgParser#varchar.
    def enterVarchar(self, ctx:apgParser.VarcharContext):
        pass

    # Exit a parse tree produced by apgParser#varchar.
    def exitVarchar(self, ctx:apgParser.VarcharContext):
        pass


    # Enter a parse tree produced by apgParser#column_option_list.
    def enterColumn_option_list(self, ctx:apgParser.Column_option_listContext):
        pass

    # Exit a parse tree produced by apgParser#column_option_list.
    def exitColumn_option_list(self, ctx:apgParser.Column_option_listContext):
        pass


    # Enter a parse tree produced by apgParser#column_option.
    def enterColumn_option(self, ctx:apgParser.Column_optionContext):
        pass

    # Exit a parse tree produced by apgParser#column_option.
    def exitColumn_option(self, ctx:apgParser.Column_optionContext):
        pass


    # Enter a parse tree produced by apgParser#primary_key.
    def enterPrimary_key(self, ctx:apgParser.Primary_keyContext):
        pass

    # Exit a parse tree produced by apgParser#primary_key.
    def exitPrimary_key(self, ctx:apgParser.Primary_keyContext):
        pass


    # Enter a parse tree produced by apgParser#column_default.
    def enterColumn_default(self, ctx:apgParser.Column_defaultContext):
        pass

    # Exit a parse tree produced by apgParser#column_default.
    def exitColumn_default(self, ctx:apgParser.Column_defaultContext):
        pass


    # Enter a parse tree produced by apgParser#ref_internal.
    def enterRef_internal(self, ctx:apgParser.Ref_internalContext):
        pass

    # Exit a parse tree produced by apgParser#ref_internal.
    def exitRef_internal(self, ctx:apgParser.Ref_internalContext):
        pass


    # Enter a parse tree produced by apgParser#ref_ext.
    def enterRef_ext(self, ctx:apgParser.Ref_extContext):
        pass

    # Exit a parse tree produced by apgParser#ref_ext.
    def exitRef_ext(self, ctx:apgParser.Ref_extContext):
        pass


    # Enter a parse tree produced by apgParser#ref_name.
    def enterRef_name(self, ctx:apgParser.Ref_nameContext):
        pass

    # Exit a parse tree produced by apgParser#ref_name.
    def exitRef_name(self, ctx:apgParser.Ref_nameContext):
        pass


    # Enter a parse tree produced by apgParser#ref_type.
    def enterRef_type(self, ctx:apgParser.Ref_typeContext):
        pass

    # Exit a parse tree produced by apgParser#ref_type.
    def exitRef_type(self, ctx:apgParser.Ref_typeContext):
        pass


    # Enter a parse tree produced by apgParser#oneToOne.
    def enterOneToOne(self, ctx:apgParser.OneToOneContext):
        pass

    # Exit a parse tree produced by apgParser#oneToOne.
    def exitOneToOne(self, ctx:apgParser.OneToOneContext):
        pass


    # Enter a parse tree produced by apgParser#oneToMany.
    def enterOneToMany(self, ctx:apgParser.OneToManyContext):
        pass

    # Exit a parse tree produced by apgParser#oneToMany.
    def exitOneToMany(self, ctx:apgParser.OneToManyContext):
        pass


    # Enter a parse tree produced by apgParser#manyToOne.
    def enterManyToOne(self, ctx:apgParser.ManyToOneContext):
        pass

    # Exit a parse tree produced by apgParser#manyToOne.
    def exitManyToOne(self, ctx:apgParser.ManyToOneContext):
        pass


    # Enter a parse tree produced by apgParser#manyToMany.
    def enterManyToMany(self, ctx:apgParser.ManyToManyContext):
        pass

    # Exit a parse tree produced by apgParser#manyToMany.
    def exitManyToMany(self, ctx:apgParser.ManyToManyContext):
        pass


    # Enter a parse tree produced by apgParser#enum_name.
    def enterEnum_name(self, ctx:apgParser.Enum_nameContext):
        pass

    # Exit a parse tree produced by apgParser#enum_name.
    def exitEnum_name(self, ctx:apgParser.Enum_nameContext):
        pass


    # Enter a parse tree produced by apgParser#enum_internal.
    def enterEnum_internal(self, ctx:apgParser.Enum_internalContext):
        pass

    # Exit a parse tree produced by apgParser#enum_internal.
    def exitEnum_internal(self, ctx:apgParser.Enum_internalContext):
        pass


    # Enter a parse tree produced by apgParser#enum_ext.
    def enterEnum_ext(self, ctx:apgParser.Enum_extContext):
        pass

    # Exit a parse tree produced by apgParser#enum_ext.
    def exitEnum_ext(self, ctx:apgParser.Enum_extContext):
        pass


    # Enter a parse tree produced by apgParser#enum_list.
    def enterEnum_list(self, ctx:apgParser.Enum_listContext):
        pass

    # Exit a parse tree produced by apgParser#enum_list.
    def exitEnum_list(self, ctx:apgParser.Enum_listContext):
        pass


    # Enter a parse tree produced by apgParser#enum_item.
    def enterEnum_item(self, ctx:apgParser.Enum_itemContext):
        pass

    # Exit a parse tree produced by apgParser#enum_item.
    def exitEnum_item(self, ctx:apgParser.Enum_itemContext):
        pass


    # Enter a parse tree produced by apgParser#enum_idx.
    def enterEnum_idx(self, ctx:apgParser.Enum_idxContext):
        pass

    # Exit a parse tree produced by apgParser#enum_idx.
    def exitEnum_idx(self, ctx:apgParser.Enum_idxContext):
        pass


    # Enter a parse tree produced by apgParser#enum_value.
    def enterEnum_value(self, ctx:apgParser.Enum_valueContext):
        pass

    # Exit a parse tree produced by apgParser#enum_value.
    def exitEnum_value(self, ctx:apgParser.Enum_valueContext):
        pass


    # Enter a parse tree produced by apgParser#check.
    def enterCheck(self, ctx:apgParser.CheckContext):
        pass

    # Exit a parse tree produced by apgParser#check.
    def exitCheck(self, ctx:apgParser.CheckContext):
        pass


    # Enter a parse tree produced by apgParser#check_expr.
    def enterCheck_expr(self, ctx:apgParser.Check_exprContext):
        pass

    # Exit a parse tree produced by apgParser#check_expr.
    def exitCheck_expr(self, ctx:apgParser.Check_exprContext):
        pass


    # Enter a parse tree produced by apgParser#layout.
    def enterLayout(self, ctx:apgParser.LayoutContext):
        pass

    # Exit a parse tree produced by apgParser#layout.
    def exitLayout(self, ctx:apgParser.LayoutContext):
        pass


    # Enter a parse tree produced by apgParser#note_option.
    def enterNote_option(self, ctx:apgParser.Note_optionContext):
        pass

    # Exit a parse tree produced by apgParser#note_option.
    def exitNote_option(self, ctx:apgParser.Note_optionContext):
        pass


    # Enter a parse tree produced by apgParser#note_value.
    def enterNote_value(self, ctx:apgParser.Note_valueContext):
        pass

    # Exit a parse tree produced by apgParser#note_value.
    def exitNote_value(self, ctx:apgParser.Note_valueContext):
        pass


    # Enter a parse tree produced by apgParser#dbview.
    def enterDbview(self, ctx:apgParser.DbviewContext):
        pass

    # Exit a parse tree produced by apgParser#dbview.
    def exitDbview(self, ctx:apgParser.DbviewContext):
        pass


    # Enter a parse tree produced by apgParser#view_name.
    def enterView_name(self, ctx:apgParser.View_nameContext):
        pass

    # Exit a parse tree produced by apgParser#view_name.
    def exitView_name(self, ctx:apgParser.View_nameContext):
        pass


    # Enter a parse tree produced by apgParser#index_ext.
    def enterIndex_ext(self, ctx:apgParser.Index_extContext):
        pass

    # Exit a parse tree produced by apgParser#index_ext.
    def exitIndex_ext(self, ctx:apgParser.Index_extContext):
        pass


    # Enter a parse tree produced by apgParser#index_int.
    def enterIndex_int(self, ctx:apgParser.Index_intContext):
        pass

    # Exit a parse tree produced by apgParser#index_int.
    def exitIndex_int(self, ctx:apgParser.Index_intContext):
        pass


    # Enter a parse tree produced by apgParser#index_item_list.
    def enterIndex_item_list(self, ctx:apgParser.Index_item_listContext):
        pass

    # Exit a parse tree produced by apgParser#index_item_list.
    def exitIndex_item_list(self, ctx:apgParser.Index_item_listContext):
        pass


    # Enter a parse tree produced by apgParser#index_item.
    def enterIndex_item(self, ctx:apgParser.Index_itemContext):
        pass

    # Exit a parse tree produced by apgParser#index_item.
    def exitIndex_item(self, ctx:apgParser.Index_itemContext):
        pass


    # Enter a parse tree produced by apgParser#column_names.
    def enterColumn_names(self, ctx:apgParser.Column_namesContext):
        pass

    # Exit a parse tree produced by apgParser#column_names.
    def exitColumn_names(self, ctx:apgParser.Column_namesContext):
        pass


    # Enter a parse tree produced by apgParser#index_name.
    def enterIndex_name(self, ctx:apgParser.Index_nameContext):
        pass

    # Exit a parse tree produced by apgParser#index_name.
    def exitIndex_name(self, ctx:apgParser.Index_nameContext):
        pass


    # Enter a parse tree produced by apgParser#dbfunc.
    def enterDbfunc(self, ctx:apgParser.DbfuncContext):
        pass

    # Exit a parse tree produced by apgParser#dbfunc.
    def exitDbfunc(self, ctx:apgParser.DbfuncContext):
        pass


    # Enter a parse tree produced by apgParser#func_name.
    def enterFunc_name(self, ctx:apgParser.Func_nameContext):
        pass

    # Exit a parse tree produced by apgParser#func_name.
    def exitFunc_name(self, ctx:apgParser.Func_nameContext):
        pass


    # Enter a parse tree produced by apgParser#func_body.
    def enterFunc_body(self, ctx:apgParser.Func_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#func_body.
    def exitFunc_body(self, ctx:apgParser.Func_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#schedule.
    def enterSchedule(self, ctx:apgParser.ScheduleContext):
        pass

    # Exit a parse tree produced by apgParser#schedule.
    def exitSchedule(self, ctx:apgParser.ScheduleContext):
        pass


    # Enter a parse tree produced by apgParser#field.
    def enterField(self, ctx:apgParser.FieldContext):
        pass

    # Exit a parse tree produced by apgParser#field.
    def exitField(self, ctx:apgParser.FieldContext):
        pass


    # Enter a parse tree produced by apgParser#sched_range.
    def enterSched_range(self, ctx:apgParser.Sched_rangeContext):
        pass

    # Exit a parse tree produced by apgParser#sched_range.
    def exitSched_range(self, ctx:apgParser.Sched_rangeContext):
        pass


    # Enter a parse tree produced by apgParser#sched_list.
    def enterSched_list(self, ctx:apgParser.Sched_listContext):
        pass

    # Exit a parse tree produced by apgParser#sched_list.
    def exitSched_list(self, ctx:apgParser.Sched_listContext):
        pass


    # Enter a parse tree produced by apgParser#business_rule.
    def enterBusiness_rule(self, ctx:apgParser.Business_ruleContext):
        pass

    # Exit a parse tree produced by apgParser#business_rule.
    def exitBusiness_rule(self, ctx:apgParser.Business_ruleContext):
        pass


    # Enter a parse tree produced by apgParser#businessRule.
    def enterBusinessRule(self, ctx:apgParser.BusinessRuleContext):
        pass

    # Exit a parse tree produced by apgParser#businessRule.
    def exitBusinessRule(self, ctx:apgParser.BusinessRuleContext):
        pass


    # Enter a parse tree produced by apgParser#IfExpression.
    def enterIfExpression(self, ctx:apgParser.IfExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#IfExpression.
    def exitIfExpression(self, ctx:apgParser.IfExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#AtTimeExpression.
    def enterAtTimeExpression(self, ctx:apgParser.AtTimeExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#AtTimeExpression.
    def exitAtTimeExpression(self, ctx:apgParser.AtTimeExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#EveryTimeExpression.
    def enterEveryTimeExpression(self, ctx:apgParser.EveryTimeExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#EveryTimeExpression.
    def exitEveryTimeExpression(self, ctx:apgParser.EveryTimeExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#OnEventExpression.
    def enterOnEventExpression(self, ctx:apgParser.OnEventExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#OnEventExpression.
    def exitOnEventExpression(self, ctx:apgParser.OnEventExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#ifExpr.
    def enterIfExpr(self, ctx:apgParser.IfExprContext):
        pass

    # Exit a parse tree produced by apgParser#ifExpr.
    def exitIfExpr(self, ctx:apgParser.IfExprContext):
        pass


    # Enter a parse tree produced by apgParser#rule_name.
    def enterRule_name(self, ctx:apgParser.Rule_nameContext):
        pass

    # Exit a parse tree produced by apgParser#rule_name.
    def exitRule_name(self, ctx:apgParser.Rule_nameContext):
        pass


    # Enter a parse tree produced by apgParser#event_desc.
    def enterEvent_desc(self, ctx:apgParser.Event_descContext):
        pass

    # Exit a parse tree produced by apgParser#event_desc.
    def exitEvent_desc(self, ctx:apgParser.Event_descContext):
        pass


    # Enter a parse tree produced by apgParser#actionExpr.
    def enterActionExpr(self, ctx:apgParser.ActionExprContext):
        pass

    # Exit a parse tree produced by apgParser#actionExpr.
    def exitActionExpr(self, ctx:apgParser.ActionExprContext):
        pass


    # Enter a parse tree produced by apgParser#action_value.
    def enterAction_value(self, ctx:apgParser.Action_valueContext):
        pass

    # Exit a parse tree produced by apgParser#action_value.
    def exitAction_value(self, ctx:apgParser.Action_valueContext):
        pass


    # Enter a parse tree produced by apgParser#action_verb.
    def enterAction_verb(self, ctx:apgParser.Action_verbContext):
        pass

    # Exit a parse tree produced by apgParser#action_verb.
    def exitAction_verb(self, ctx:apgParser.Action_verbContext):
        pass


    # Enter a parse tree produced by apgParser#action_object.
    def enterAction_object(self, ctx:apgParser.Action_objectContext):
        pass

    # Exit a parse tree produced by apgParser#action_object.
    def exitAction_object(self, ctx:apgParser.Action_objectContext):
        pass


    # Enter a parse tree produced by apgParser#inverseTrigonometricSin.
    def enterInverseTrigonometricSin(self, ctx:apgParser.InverseTrigonometricSinContext):
        pass

    # Exit a parse tree produced by apgParser#inverseTrigonometricSin.
    def exitInverseTrigonometricSin(self, ctx:apgParser.InverseTrigonometricSinContext):
        pass


    # Enter a parse tree produced by apgParser#statisticalMinimum.
    def enterStatisticalMinimum(self, ctx:apgParser.StatisticalMinimumContext):
        pass

    # Exit a parse tree produced by apgParser#statisticalMinimum.
    def exitStatisticalMinimum(self, ctx:apgParser.StatisticalMinimumContext):
        pass


    # Enter a parse tree produced by apgParser#binaryAdditionSubtraction.
    def enterBinaryAdditionSubtraction(self, ctx:apgParser.BinaryAdditionSubtractionContext):
        pass

    # Exit a parse tree produced by apgParser#binaryAdditionSubtraction.
    def exitBinaryAdditionSubtraction(self, ctx:apgParser.BinaryAdditionSubtractionContext):
        pass


    # Enter a parse tree produced by apgParser#booleanCombination.
    def enterBooleanCombination(self, ctx:apgParser.BooleanCombinationContext):
        pass

    # Exit a parse tree produced by apgParser#booleanCombination.
    def exitBooleanCombination(self, ctx:apgParser.BooleanCombinationContext):
        pass


    # Enter a parse tree produced by apgParser#nestedExpr.
    def enterNestedExpr(self, ctx:apgParser.NestedExprContext):
        pass

    # Exit a parse tree produced by apgParser#nestedExpr.
    def exitNestedExpr(self, ctx:apgParser.NestedExprContext):
        pass


    # Enter a parse tree produced by apgParser#trigonometricCos.
    def enterTrigonometricCos(self, ctx:apgParser.TrigonometricCosContext):
        pass

    # Exit a parse tree produced by apgParser#trigonometricCos.
    def exitTrigonometricCos(self, ctx:apgParser.TrigonometricCosContext):
        pass


    # Enter a parse tree produced by apgParser#inverseTrigonometricTan.
    def enterInverseTrigonometricTan(self, ctx:apgParser.InverseTrigonometricTanContext):
        pass

    # Exit a parse tree produced by apgParser#inverseTrigonometricTan.
    def exitInverseTrigonometricTan(self, ctx:apgParser.InverseTrigonometricTanContext):
        pass


    # Enter a parse tree produced by apgParser#notExpression.
    def enterNotExpression(self, ctx:apgParser.NotExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#notExpression.
    def exitNotExpression(self, ctx:apgParser.NotExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#inverseHyperbolicTangent.
    def enterInverseHyperbolicTangent(self, ctx:apgParser.InverseHyperbolicTangentContext):
        pass

    # Exit a parse tree produced by apgParser#inverseHyperbolicTangent.
    def exitInverseHyperbolicTangent(self, ctx:apgParser.InverseHyperbolicTangentContext):
        pass


    # Enter a parse tree produced by apgParser#binaryComparison.
    def enterBinaryComparison(self, ctx:apgParser.BinaryComparisonContext):
        pass

    # Exit a parse tree produced by apgParser#binaryComparison.
    def exitBinaryComparison(self, ctx:apgParser.BinaryComparisonContext):
        pass


    # Enter a parse tree produced by apgParser#trigonometricTan.
    def enterTrigonometricTan(self, ctx:apgParser.TrigonometricTanContext):
        pass

    # Exit a parse tree produced by apgParser#trigonometricTan.
    def exitTrigonometricTan(self, ctx:apgParser.TrigonometricTanContext):
        pass


    # Enter a parse tree produced by apgParser#literalExpr.
    def enterLiteralExpr(self, ctx:apgParser.LiteralExprContext):
        pass

    # Exit a parse tree produced by apgParser#literalExpr.
    def exitLiteralExpr(self, ctx:apgParser.LiteralExprContext):
        pass


    # Enter a parse tree produced by apgParser#functionCallExpr.
    def enterFunctionCallExpr(self, ctx:apgParser.FunctionCallExprContext):
        pass

    # Exit a parse tree produced by apgParser#functionCallExpr.
    def exitFunctionCallExpr(self, ctx:apgParser.FunctionCallExprContext):
        pass


    # Enter a parse tree produced by apgParser#statisticalAverage.
    def enterStatisticalAverage(self, ctx:apgParser.StatisticalAverageContext):
        pass

    # Exit a parse tree produced by apgParser#statisticalAverage.
    def exitStatisticalAverage(self, ctx:apgParser.StatisticalAverageContext):
        pass


    # Enter a parse tree produced by apgParser#binaryMultiplicationDiv.
    def enterBinaryMultiplicationDiv(self, ctx:apgParser.BinaryMultiplicationDivContext):
        pass

    # Exit a parse tree produced by apgParser#binaryMultiplicationDiv.
    def exitBinaryMultiplicationDiv(self, ctx:apgParser.BinaryMultiplicationDivContext):
        pass


    # Enter a parse tree produced by apgParser#hyperbolicCosine.
    def enterHyperbolicCosine(self, ctx:apgParser.HyperbolicCosineContext):
        pass

    # Exit a parse tree produced by apgParser#hyperbolicCosine.
    def exitHyperbolicCosine(self, ctx:apgParser.HyperbolicCosineContext):
        pass


    # Enter a parse tree produced by apgParser#statisticalMaximum.
    def enterStatisticalMaximum(self, ctx:apgParser.StatisticalMaximumContext):
        pass

    # Exit a parse tree produced by apgParser#statisticalMaximum.
    def exitStatisticalMaximum(self, ctx:apgParser.StatisticalMaximumContext):
        pass


    # Enter a parse tree produced by apgParser#trigonometricSin.
    def enterTrigonometricSin(self, ctx:apgParser.TrigonometricSinContext):
        pass

    # Exit a parse tree produced by apgParser#trigonometricSin.
    def exitTrigonometricSin(self, ctx:apgParser.TrigonometricSinContext):
        pass


    # Enter a parse tree produced by apgParser#inverseHyperbolicCosine.
    def enterInverseHyperbolicCosine(self, ctx:apgParser.InverseHyperbolicCosineContext):
        pass

    # Exit a parse tree produced by apgParser#inverseHyperbolicCosine.
    def exitInverseHyperbolicCosine(self, ctx:apgParser.InverseHyperbolicCosineContext):
        pass


    # Enter a parse tree produced by apgParser#inverseHyperbolicSine.
    def enterInverseHyperbolicSine(self, ctx:apgParser.InverseHyperbolicSineContext):
        pass

    # Exit a parse tree produced by apgParser#inverseHyperbolicSine.
    def exitInverseHyperbolicSine(self, ctx:apgParser.InverseHyperbolicSineContext):
        pass


    # Enter a parse tree produced by apgParser#statisticalSum.
    def enterStatisticalSum(self, ctx:apgParser.StatisticalSumContext):
        pass

    # Exit a parse tree produced by apgParser#statisticalSum.
    def exitStatisticalSum(self, ctx:apgParser.StatisticalSumContext):
        pass


    # Enter a parse tree produced by apgParser#hyperbolicTangent.
    def enterHyperbolicTangent(self, ctx:apgParser.HyperbolicTangentContext):
        pass

    # Exit a parse tree produced by apgParser#hyperbolicTangent.
    def exitHyperbolicTangent(self, ctx:apgParser.HyperbolicTangentContext):
        pass


    # Enter a parse tree produced by apgParser#inverseTrigonometricCos.
    def enterInverseTrigonometricCos(self, ctx:apgParser.InverseTrigonometricCosContext):
        pass

    # Exit a parse tree produced by apgParser#inverseTrigonometricCos.
    def exitInverseTrigonometricCos(self, ctx:apgParser.InverseTrigonometricCosContext):
        pass


    # Enter a parse tree produced by apgParser#hyperbolicSine.
    def enterHyperbolicSine(self, ctx:apgParser.HyperbolicSineContext):
        pass

    # Exit a parse tree produced by apgParser#hyperbolicSine.
    def exitHyperbolicSine(self, ctx:apgParser.HyperbolicSineContext):
        pass


    # Enter a parse tree produced by apgParser#unaryMinus.
    def enterUnaryMinus(self, ctx:apgParser.UnaryMinusContext):
        pass

    # Exit a parse tree produced by apgParser#unaryMinus.
    def exitUnaryMinus(self, ctx:apgParser.UnaryMinusContext):
        pass


    # Enter a parse tree produced by apgParser#identExpression.
    def enterIdentExpression(self, ctx:apgParser.IdentExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#identExpression.
    def exitIdentExpression(self, ctx:apgParser.IdentExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#identFunction.
    def enterIdentFunction(self, ctx:apgParser.IdentFunctionContext):
        pass

    # Exit a parse tree produced by apgParser#identFunction.
    def exitIdentFunction(self, ctx:apgParser.IdentFunctionContext):
        pass


    # Enter a parse tree produced by apgParser#expr_list.
    def enterExpr_list(self, ctx:apgParser.Expr_listContext):
        pass

    # Exit a parse tree produced by apgParser#expr_list.
    def exitExpr_list(self, ctx:apgParser.Expr_listContext):
        pass


    # Enter a parse tree produced by apgParser#literal.
    def enterLiteral(self, ctx:apgParser.LiteralContext):
        pass

    # Exit a parse tree produced by apgParser#literal.
    def exitLiteral(self, ctx:apgParser.LiteralContext):
        pass


    # Enter a parse tree produced by apgParser#booleanOp.
    def enterBooleanOp(self, ctx:apgParser.BooleanOpContext):
        pass

    # Exit a parse tree produced by apgParser#booleanOp.
    def exitBooleanOp(self, ctx:apgParser.BooleanOpContext):
        pass


    # Enter a parse tree produced by apgParser#comparisonOp.
    def enterComparisonOp(self, ctx:apgParser.ComparisonOpContext):
        pass

    # Exit a parse tree produced by apgParser#comparisonOp.
    def exitComparisonOp(self, ctx:apgParser.ComparisonOpContext):
        pass


    # Enter a parse tree produced by apgParser#functionCall.
    def enterFunctionCall(self, ctx:apgParser.FunctionCallContext):
        pass

    # Exit a parse tree produced by apgParser#functionCall.
    def exitFunctionCall(self, ctx:apgParser.FunctionCallContext):
        pass


    # Enter a parse tree produced by apgParser#function_name.
    def enterFunction_name(self, ctx:apgParser.Function_nameContext):
        pass

    # Exit a parse tree produced by apgParser#function_name.
    def exitFunction_name(self, ctx:apgParser.Function_nameContext):
        pass


    # Enter a parse tree produced by apgParser#param_list.
    def enterParam_list(self, ctx:apgParser.Param_listContext):
        pass

    # Exit a parse tree produced by apgParser#param_list.
    def exitParam_list(self, ctx:apgParser.Param_listContext):
        pass


    # Enter a parse tree produced by apgParser#trigger.
    def enterTrigger(self, ctx:apgParser.TriggerContext):
        pass

    # Exit a parse tree produced by apgParser#trigger.
    def exitTrigger(self, ctx:apgParser.TriggerContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_name.
    def enterTrigger_name(self, ctx:apgParser.Trigger_nameContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_name.
    def exitTrigger_name(self, ctx:apgParser.Trigger_nameContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_event.
    def enterTrigger_event(self, ctx:apgParser.Trigger_eventContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_event.
    def exitTrigger_event(self, ctx:apgParser.Trigger_eventContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_statement.
    def enterTrigger_statement(self, ctx:apgParser.Trigger_statementContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_statement.
    def exitTrigger_statement(self, ctx:apgParser.Trigger_statementContext):
        pass


    # Enter a parse tree produced by apgParser#sql_stmt_list.
    def enterSql_stmt_list(self, ctx:apgParser.Sql_stmt_listContext):
        pass

    # Exit a parse tree produced by apgParser#sql_stmt_list.
    def exitSql_stmt_list(self, ctx:apgParser.Sql_stmt_listContext):
        pass


    # Enter a parse tree produced by apgParser#sql_stmt.
    def enterSql_stmt(self, ctx:apgParser.Sql_stmtContext):
        pass

    # Exit a parse tree produced by apgParser#sql_stmt.
    def exitSql_stmt(self, ctx:apgParser.Sql_stmtContext):
        pass


    # Enter a parse tree produced by apgParser#column_assignment.
    def enterColumn_assignment(self, ctx:apgParser.Column_assignmentContext):
        pass

    # Exit a parse tree produced by apgParser#column_assignment.
    def exitColumn_assignment(self, ctx:apgParser.Column_assignmentContext):
        pass


    # Enter a parse tree produced by apgParser#script.
    def enterScript(self, ctx:apgParser.ScriptContext):
        pass

    # Exit a parse tree produced by apgParser#script.
    def exitScript(self, ctx:apgParser.ScriptContext):
        pass


    # Enter a parse tree produced by apgParser#script_lang.
    def enterScript_lang(self, ctx:apgParser.Script_langContext):
        pass

    # Exit a parse tree produced by apgParser#script_lang.
    def exitScript_lang(self, ctx:apgParser.Script_langContext):
        pass


    # Enter a parse tree produced by apgParser#script_body.
    def enterScript_body(self, ctx:apgParser.Script_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#script_body.
    def exitScript_body(self, ctx:apgParser.Script_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#script_name.
    def enterScript_name(self, ctx:apgParser.Script_nameContext):
        pass

    # Exit a parse tree produced by apgParser#script_name.
    def exitScript_name(self, ctx:apgParser.Script_nameContext):
        pass


    # Enter a parse tree produced by apgParser#workflow.
    def enterWorkflow(self, ctx:apgParser.WorkflowContext):
        pass

    # Exit a parse tree produced by apgParser#workflow.
    def exitWorkflow(self, ctx:apgParser.WorkflowContext):
        pass


    # Enter a parse tree produced by apgParser#workflow_name.
    def enterWorkflow_name(self, ctx:apgParser.Workflow_nameContext):
        pass

    # Exit a parse tree produced by apgParser#workflow_name.
    def exitWorkflow_name(self, ctx:apgParser.Workflow_nameContext):
        pass


    # Enter a parse tree produced by apgParser#workflow_step.
    def enterWorkflow_step(self, ctx:apgParser.Workflow_stepContext):
        pass

    # Exit a parse tree produced by apgParser#workflow_step.
    def exitWorkflow_step(self, ctx:apgParser.Workflow_stepContext):
        pass


    # Enter a parse tree produced by apgParser#step_name.
    def enterStep_name(self, ctx:apgParser.Step_nameContext):
        pass

    # Exit a parse tree produced by apgParser#step_name.
    def exitStep_name(self, ctx:apgParser.Step_nameContext):
        pass


    # Enter a parse tree produced by apgParser#workflow_statement.
    def enterWorkflow_statement(self, ctx:apgParser.Workflow_statementContext):
        pass

    # Exit a parse tree produced by apgParser#workflow_statement.
    def exitWorkflow_statement(self, ctx:apgParser.Workflow_statementContext):
        pass



del apgParser