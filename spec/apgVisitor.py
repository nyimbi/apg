# Generated from /Users/nyimbiodero/src/pjs/apg/spec/apg.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .apgParser import apgParser
else:
    from apgParser import apgParser

# This class defines a complete generic visitor for a parse tree produced by apgParser.

class apgVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by apgParser#program.
    def visitProgram(self, ctx:apgParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#module_declaration.
    def visitModule_declaration(self, ctx:apgParser.Module_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#module_name.
    def visitModule_name(self, ctx:apgParser.Module_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#module_metadata.
    def visitModule_metadata(self, ctx:apgParser.Module_metadataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#module_property.
    def visitModule_property(self, ctx:apgParser.Module_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dependency_list.
    def visitDependency_list(self, ctx:apgParser.Dependency_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dependency.
    def visitDependency(self, ctx:apgParser.DependencyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#version_constraint.
    def visitVersion_constraint(self, ctx:apgParser.Version_constraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#version_range.
    def visitVersion_range(self, ctx:apgParser.Version_rangeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#export_list.
    def visitExport_list(self, ctx:apgParser.Export_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#export_item.
    def visitExport_item(self, ctx:apgParser.Export_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#private_list.
    def visitPrivate_list(self, ctx:apgParser.Private_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alias.
    def visitAlias(self, ctx:apgParser.AliasContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#import_statement.
    def visitImport_statement(self, ctx:apgParser.Import_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#include_statement.
    def visitInclude_statement(self, ctx:apgParser.Include_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#export_statement.
    def visitExport_statement(self, ctx:apgParser.Export_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#export_declaration.
    def visitExport_declaration(self, ctx:apgParser.Export_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#module_path.
    def visitModule_path(self, ctx:apgParser.Module_pathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#file_path.
    def visitFile_path(self, ctx:apgParser.File_pathContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#import_options.
    def visitImport_options(self, ctx:apgParser.Import_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#import_option.
    def visitImport_option(self, ctx:apgParser.Import_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#include_options.
    def visitInclude_options(self, ctx:apgParser.Include_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#include_option.
    def visitInclude_option(self, ctx:apgParser.Include_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#import_list.
    def visitImport_list(self, ctx:apgParser.Import_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#import_item.
    def visitImport_item(self, ctx:apgParser.Import_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#entity.
    def visitEntity(self, ctx:apgParser.EntityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#decorator.
    def visitDecorator(self, ctx:apgParser.DecoratorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#entity_type.
    def visitEntity_type(self, ctx:apgParser.Entity_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inheritance.
    def visitInheritance(self, ctx:apgParser.InheritanceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#version_tag.
    def visitVersion_tag(self, ctx:apgParser.Version_tagContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#entity_body.
    def visitEntity_body(self, ctx:apgParser.Entity_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#entity_member.
    def visitEntity_member(self, ctx:apgParser.Entity_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#config_item.
    def visitConfig_item(self, ctx:apgParser.Config_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#capability_contract_block.
    def visitCapability_contract_block(self, ctx:apgParser.Capability_contract_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#capability_contract.
    def visitCapability_contract(self, ctx:apgParser.Capability_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#capability_contract_member.
    def visitCapability_contract_member(self, ctx:apgParser.Capability_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_component_block.
    def visitErp_component_block(self, ctx:apgParser.Erp_component_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_module_set.
    def visitErp_module_set(self, ctx:apgParser.Erp_module_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_component_set.
    def visitErp_component_set(self, ctx:apgParser.Erp_component_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_component_binding.
    def visitErp_component_binding(self, ctx:apgParser.Erp_component_bindingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_component_key.
    def visitErp_component_key(self, ctx:apgParser.Erp_component_keyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_component_ref.
    def visitErp_component_ref(self, ctx:apgParser.Erp_component_refContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_domain.
    def visitErp_domain(self, ctx:apgParser.Erp_domainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_component_member.
    def visitErp_component_member(self, ctx:apgParser.Erp_component_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_data_contract.
    def visitErp_data_contract(self, ctx:apgParser.Erp_data_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_data_member.
    def visitErp_data_member(self, ctx:apgParser.Erp_data_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_api_contract.
    def visitErp_api_contract(self, ctx:apgParser.Erp_api_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_api_member.
    def visitErp_api_member(self, ctx:apgParser.Erp_api_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_workflow_contract.
    def visitErp_workflow_contract(self, ctx:apgParser.Erp_workflow_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_workflow_member.
    def visitErp_workflow_member(self, ctx:apgParser.Erp_workflow_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_rule_set.
    def visitErp_rule_set(self, ctx:apgParser.Erp_rule_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#erp_rule_group.
    def visitErp_rule_group(self, ctx:apgParser.Erp_rule_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#approval_contract.
    def visitApproval_contract(self, ctx:apgParser.Approval_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#approval_member.
    def visitApproval_member(self, ctx:apgParser.Approval_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#permission_contract.
    def visitPermission_contract(self, ctx:apgParser.Permission_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#permission_member.
    def visitPermission_member(self, ctx:apgParser.Permission_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#audit_contract.
    def visitAudit_contract(self, ctx:apgParser.Audit_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#audit_member.
    def visitAudit_member(self, ctx:apgParser.Audit_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#effective_date_contract.
    def visitEffective_date_contract(self, ctx:apgParser.Effective_date_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#effective_date_member.
    def visitEffective_date_member(self, ctx:apgParser.Effective_date_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#master_data_contract.
    def visitMaster_data_contract(self, ctx:apgParser.Master_data_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#master_data_member.
    def visitMaster_data_member(self, ctx:apgParser.Master_data_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_composition_block.
    def visitAgent_composition_block(self, ctx:apgParser.Agent_composition_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_set.
    def visitAgent_set(self, ctx:apgParser.Agent_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_binding.
    def visitAgent_binding(self, ctx:apgParser.Agent_bindingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_contract_member.
    def visitAgent_contract_member(self, ctx:apgParser.Agent_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_runtime_set.
    def visitAgent_runtime_set(self, ctx:apgParser.Agent_runtime_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_runtime_contract_member.
    def visitAgent_runtime_contract_member(self, ctx:apgParser.Agent_runtime_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_runtime_ref.
    def visitAgent_runtime_ref(self, ctx:apgParser.Agent_runtime_refContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_tool_set.
    def visitAgent_tool_set(self, ctx:apgParser.Agent_tool_setContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_tool_contract_member.
    def visitAgent_tool_contract_member(self, ctx:apgParser.Agent_tool_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_memory_contract.
    def visitAgent_memory_contract(self, ctx:apgParser.Agent_memory_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#handoff_graph.
    def visitHandoff_graph(self, ctx:apgParser.Handoff_graphContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#handoff_edge.
    def visitHandoff_edge(self, ctx:apgParser.Handoff_edgeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#handoff_modifier.
    def visitHandoff_modifier(self, ctx:apgParser.Handoff_modifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#model_chain.
    def visitModel_chain(self, ctx:apgParser.Model_chainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#model_ref.
    def visitModel_ref(self, ctx:apgParser.Model_refContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#capability_ref.
    def visitCapability_ref(self, ctx:apgParser.Capability_refContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#io_contract.
    def visitIo_contract(self, ctx:apgParser.Io_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_engine_block.
    def visitRule_engine_block(self, ctx:apgParser.Rule_engine_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_engine_contract.
    def visitRule_engine_contract(self, ctx:apgParser.Rule_engine_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_engine_member.
    def visitRule_engine_member(self, ctx:apgParser.Rule_engine_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_engine_type.
    def visitRule_engine_type(self, ctx:apgParser.Rule_engine_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_list.
    def visitRule_list(self, ctx:apgParser.Rule_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_contract.
    def visitRule_contract(self, ctx:apgParser.Rule_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_contract_member.
    def visitRule_contract_member(self, ctx:apgParser.Rule_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rule_decision.
    def visitRule_decision(self, ctx:apgParser.Rule_decisionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_contract_block.
    def visitUi_contract_block(self, ctx:apgParser.Ui_contract_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_contract.
    def visitUi_contract(self, ctx:apgParser.Ui_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_contract_member.
    def visitUi_contract_member(self, ctx:apgParser.Ui_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_shell.
    def visitUi_shell(self, ctx:apgParser.Ui_shellContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_route_list.
    def visitUi_route_list(self, ctx:apgParser.Ui_route_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_route.
    def visitUi_route(self, ctx:apgParser.Ui_routeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ui_route_member.
    def visitUi_route_member(self, ctx:apgParser.Ui_route_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_contract_block.
    def visitTheme_contract_block(self, ctx:apgParser.Theme_contract_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_contract.
    def visitTheme_contract(self, ctx:apgParser.Theme_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_contract_member.
    def visitTheme_contract_member(self, ctx:apgParser.Theme_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_token_map.
    def visitTheme_token_map(self, ctx:apgParser.Theme_token_mapContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_token.
    def visitTheme_token(self, ctx:apgParser.Theme_tokenContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#runtime_contract.
    def visitRuntime_contract(self, ctx:apgParser.Runtime_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#runtime_contract_member.
    def visitRuntime_contract_member(self, ctx:apgParser.Runtime_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#runtime_backend.
    def visitRuntime_backend(self, ctx:apgParser.Runtime_backendContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#stream_runtime_block.
    def visitStream_runtime_block(self, ctx:apgParser.Stream_runtime_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#stream_runtime_contract.
    def visitStream_runtime_contract(self, ctx:apgParser.Stream_runtime_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#stream_runtime_member.
    def visitStream_runtime_member(self, ctx:apgParser.Stream_runtime_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#stream_processor.
    def visitStream_processor(self, ctx:apgParser.Stream_processorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#i18n_contract_block.
    def visitI18n_contract_block(self, ctx:apgParser.I18n_contract_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#i18n_contract.
    def visitI18n_contract(self, ctx:apgParser.I18n_contractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#i18n_contract_member.
    def visitI18n_contract_member(self, ctx:apgParser.I18n_contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#language_collection.
    def visitLanguage_collection(self, ctx:apgParser.Language_collectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#language_list.
    def visitLanguage_list(self, ctx:apgParser.Language_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#language_code.
    def visitLanguage_code(self, ctx:apgParser.Language_codeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_list.
    def visitReference_list(self, ctx:apgParser.Reference_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#contract_object.
    def visitContract_object(self, ctx:apgParser.Contract_objectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#contract_member.
    def visitContract_member(self, ctx:apgParser.Contract_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#contract_value.
    def visitContract_value(self, ctx:apgParser.Contract_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#contract_array.
    def visitContract_array(self, ctx:apgParser.Contract_arrayContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#contract_scalar.
    def visitContract_scalar(self, ctx:apgParser.Contract_scalarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#contract_separator.
    def visitContract_separator(self, ctx:apgParser.Contract_separatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#type_annotation.
    def visitType_annotation(self, ctx:apgParser.Type_annotationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#union_type.
    def visitUnion_type(self, ctx:apgParser.Union_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#primary_type.
    def visitPrimary_type(self, ctx:apgParser.Primary_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#basic_type.
    def visitBasic_type(self, ctx:apgParser.Basic_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#optional_suffix.
    def visitOptional_suffix(self, ctx:apgParser.Optional_suffixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#generic_type.
    def visitGeneric_type(self, ctx:apgParser.Generic_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#list_type.
    def visitList_type(self, ctx:apgParser.List_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dict_type.
    def visitDict_type(self, ctx:apgParser.Dict_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#value_expr.
    def visitValue_expr(self, ctx:apgParser.Value_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#simple_value.
    def visitSimple_value(self, ctx:apgParser.Simple_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#env_var.
    def visitEnv_var(self, ctx:apgParser.Env_varContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#f_string.
    def visitF_string(self, ctx:apgParser.F_stringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#list_value.
    def visitList_value(self, ctx:apgParser.List_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dict_value.
    def visitDict_value(self, ctx:apgParser.Dict_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#key_value_pair.
    def visitKey_value_pair(self, ctx:apgParser.Key_value_pairContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cascade_value.
    def visitCascade_value(self, ctx:apgParser.Cascade_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agent_memory_value.
    def visitAgent_memory_value(self, ctx:apgParser.Agent_memory_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_value.
    def visitReference_value(self, ctx:apgParser.Reference_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#combination_expr.
    def visitCombination_expr(self, ctx:apgParser.Combination_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#url_pattern.
    def visitUrl_pattern(self, ctx:apgParser.Url_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#regex_pattern.
    def visitRegex_pattern(self, ctx:apgParser.Regex_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#time_expr.
    def visitTime_expr(self, ctx:apgParser.Time_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#async_expr.
    def visitAsync_expr(self, ctx:apgParser.Async_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#behavior_item.
    def visitBehavior_item(self, ctx:apgParser.Behavior_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#annotation.
    def visitAnnotation(self, ctx:apgParser.AnnotationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#annotation_body.
    def visitAnnotation_body(self, ctx:apgParser.Annotation_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#annotation_member.
    def visitAnnotation_member(self, ctx:apgParser.Annotation_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#nested_annotation.
    def visitNested_annotation(self, ctx:apgParser.Nested_annotationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#when_clause.
    def visitWhen_clause(self, ctx:apgParser.When_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#then_clause.
    def visitThen_clause(self, ctx:apgParser.Then_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#method_def.
    def visitMethod_def(self, ctx:apgParser.Method_defContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#async_modifier.
    def visitAsync_modifier(self, ctx:apgParser.Async_modifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#param_list.
    def visitParam_list(self, ctx:apgParser.Param_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#parameter.
    def visitParameter(self, ctx:apgParser.ParameterContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#return_type.
    def visitReturn_type(self, ctx:apgParser.Return_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#method_body.
    def visitMethod_body(self, ctx:apgParser.Method_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statement.
    def visitStatement(self, ctx:apgParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#simple_statement.
    def visitSimple_statement(self, ctx:apgParser.Simple_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#compound_statement.
    def visitCompound_statement(self, ctx:apgParser.Compound_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#assignment.
    def visitAssignment(self, ctx:apgParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#method_call.
    def visitMethod_call(self, ctx:apgParser.Method_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#args.
    def visitArgs(self, ctx:apgParser.ArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#argument.
    def visitArgument(self, ctx:apgParser.ArgumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#return_statement.
    def visitReturn_statement(self, ctx:apgParser.Return_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#break_statement.
    def visitBreak_statement(self, ctx:apgParser.Break_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#continue_statement.
    def visitContinue_statement(self, ctx:apgParser.Continue_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#pass_statement.
    def visitPass_statement(self, ctx:apgParser.Pass_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#assert_statement.
    def visitAssert_statement(self, ctx:apgParser.Assert_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#yield_statement.
    def visitYield_statement(self, ctx:apgParser.Yield_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#if_statement.
    def visitIf_statement(self, ctx:apgParser.If_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#elif_clause.
    def visitElif_clause(self, ctx:apgParser.Elif_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#else_clause.
    def visitElse_clause(self, ctx:apgParser.Else_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#for_statement.
    def visitFor_statement(self, ctx:apgParser.For_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#while_statement.
    def visitWhile_statement(self, ctx:apgParser.While_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#try_statement.
    def visitTry_statement(self, ctx:apgParser.Try_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#except_clause.
    def visitExcept_clause(self, ctx:apgParser.Except_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exception_spec.
    def visitException_spec(self, ctx:apgParser.Exception_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#finally_clause.
    def visitFinally_clause(self, ctx:apgParser.Finally_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#with_statement.
    def visitWith_statement(self, ctx:apgParser.With_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#with_item.
    def visitWith_item(self, ctx:apgParser.With_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#match_statement.
    def visitMatch_statement(self, ctx:apgParser.Match_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#case_clause.
    def visitCase_clause(self, ctx:apgParser.Case_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#pattern.
    def visitPattern(self, ctx:apgParser.PatternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#or_pattern.
    def visitOr_pattern(self, ctx:apgParser.Or_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#primary_pattern.
    def visitPrimary_pattern(self, ctx:apgParser.Primary_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#literal_pattern.
    def visitLiteral_pattern(self, ctx:apgParser.Literal_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#capture_pattern.
    def visitCapture_pattern(self, ctx:apgParser.Capture_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#wildcard_pattern.
    def visitWildcard_pattern(self, ctx:apgParser.Wildcard_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#value_pattern.
    def visitValue_pattern(self, ctx:apgParser.Value_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sequence_pattern.
    def visitSequence_pattern(self, ctx:apgParser.Sequence_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mapping_pattern.
    def visitMapping_pattern(self, ctx:apgParser.Mapping_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mapping_pattern_pair.
    def visitMapping_pattern_pair(self, ctx:apgParser.Mapping_pattern_pairContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#class_pattern.
    def visitClass_pattern(self, ctx:apgParser.Class_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#guard.
    def visitGuard(self, ctx:apgParser.GuardContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#async_statement.
    def visitAsync_statement(self, ctx:apgParser.Async_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#statement_block.
    def visitStatement_block(self, ctx:apgParser.Statement_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#expression.
    def visitExpression(self, ctx:apgParser.ExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#lambda_expr.
    def visitLambda_expr(self, ctx:apgParser.Lambda_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#conditional_expr.
    def visitConditional_expr(self, ctx:apgParser.Conditional_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#or_test.
    def visitOr_test(self, ctx:apgParser.Or_testContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#and_test.
    def visitAnd_test(self, ctx:apgParser.And_testContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#not_test.
    def visitNot_test(self, ctx:apgParser.Not_testContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#comparison.
    def visitComparison(self, ctx:apgParser.ComparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#comp_op.
    def visitComp_op(self, ctx:apgParser.Comp_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bitwise_or.
    def visitBitwise_or(self, ctx:apgParser.Bitwise_orContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bitwise_xor.
    def visitBitwise_xor(self, ctx:apgParser.Bitwise_xorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bitwise_and.
    def visitBitwise_and(self, ctx:apgParser.Bitwise_andContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#shift_expr.
    def visitShift_expr(self, ctx:apgParser.Shift_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#arith_expr.
    def visitArith_expr(self, ctx:apgParser.Arith_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#term.
    def visitTerm(self, ctx:apgParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#factor.
    def visitFactor(self, ctx:apgParser.FactorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#power.
    def visitPower(self, ctx:apgParser.PowerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#atom_expr.
    def visitAtom_expr(self, ctx:apgParser.Atom_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#atom.
    def visitAtom(self, ctx:apgParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#await_expr.
    def visitAwait_expr(self, ctx:apgParser.Await_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trailer.
    def visitTrailer(self, ctx:apgParser.TrailerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#subscriptlist.
    def visitSubscriptlist(self, ctx:apgParser.SubscriptlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#subscript.
    def visitSubscript(self, ctx:apgParser.SubscriptContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sliceop.
    def visitSliceop(self, ctx:apgParser.SliceopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#listmaker.
    def visitListmaker(self, ctx:apgParser.ListmakerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dictorsetmaker.
    def visitDictorsetmaker(self, ctx:apgParser.DictorsetmakerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#testlist_comp.
    def visitTestlist_comp(self, ctx:apgParser.Testlist_compContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#star_expr.
    def visitStar_expr(self, ctx:apgParser.Star_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#comp_for.
    def visitComp_for(self, ctx:apgParser.Comp_forContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#list_for.
    def visitList_for(self, ctx:apgParser.List_forContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exprlist.
    def visitExprlist(self, ctx:apgParser.ExprlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#testlist.
    def visitTestlist(self, ctx:apgParser.TestlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#yield_expr.
    def visitYield_expr(self, ctx:apgParser.Yield_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#yield_arg.
    def visitYield_arg(self, ctx:apgParser.Yield_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#flow_definition.
    def visitFlow_definition(self, ctx:apgParser.Flow_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#flow_step.
    def visitFlow_step(self, ctx:apgParser.Flow_stepContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#flow_connector.
    def visitFlow_connector(self, ctx:apgParser.Flow_connectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#flow_modifiers.
    def visitFlow_modifiers(self, ctx:apgParser.Flow_modifiersContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#flow_modifier.
    def visitFlow_modifier(self, ctx:apgParser.Flow_modifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#conditional_flow_step.
    def visitConditional_flow_step(self, ctx:apgParser.Conditional_flow_stepContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#parallel_flow_step.
    def visitParallel_flow_step(self, ctx:apgParser.Parallel_flow_stepContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#minion_command.
    def visitMinion_command(self, ctx:apgParser.Minion_commandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#broadcast_command.
    def visitBroadcast_command(self, ctx:apgParser.Broadcast_commandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#minion_verb.
    def visitMinion_verb(self, ctx:apgParser.Minion_verbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#minion_scope.
    def visitMinion_scope(self, ctx:apgParser.Minion_scopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#nested_entity.
    def visitNested_entity(self, ctx:apgParser.Nested_entityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#class_def.
    def visitClass_def(self, ctx:apgParser.Class_defContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#class_body.
    def visitClass_body(self, ctx:apgParser.Class_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#class_member.
    def visitClass_member(self, ctx:apgParser.Class_memberContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exception_def.
    def visitException_def(self, ctx:apgParser.Exception_defContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exception_body.
    def visitException_body(self, ctx:apgParser.Exception_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#variable_declaration.
    def visitVariable_declaration(self, ctx:apgParser.Variable_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ab_testing_config.
    def visitAb_testing_config(self, ctx:apgParser.Ab_testing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#access_policy_specification.
    def visitAccess_policy_specification(self, ctx:apgParser.Access_policy_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#aggregation_specification.
    def visitAggregation_specification(self, ctx:apgParser.Aggregation_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alert_frequency.
    def visitAlert_frequency(self, ctx:apgParser.Alert_frequencyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alerting_specification.
    def visitAlerting_specification(self, ctx:apgParser.Alerting_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#algorithm_specification.
    def visitAlgorithm_specification(self, ctx:apgParser.Algorithm_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#analytics_specification.
    def visitAnalytics_specification(self, ctx:apgParser.Analytics_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#announcement_config.
    def visitAnnouncement_config(self, ctx:apgParser.Announcement_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#api_call.
    def visitApi_call(self, ctx:apgParser.Api_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#api_gateway_config.
    def visitApi_gateway_config(self, ctx:apgParser.Api_gateway_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#api_specification.
    def visitApi_specification(self, ctx:apgParser.Api_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#auto_resolution_specification.
    def visitAuto_resolution_specification(self, ctx:apgParser.Auto_resolution_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#autocomplete_config.
    def visitAutocomplete_config(self, ctx:apgParser.Autocomplete_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#background_check_config.
    def visitBackground_check_config(self, ctx:apgParser.Background_check_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#backup_specification.
    def visitBackup_specification(self, ctx:apgParser.Backup_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bi_configuration.
    def visitBi_configuration(self, ctx:apgParser.Bi_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bucket_specification.
    def visitBucket_specification(self, ctx:apgParser.Bucket_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#chat_configuration.
    def visitChat_configuration(self, ctx:apgParser.Chat_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#circuit_breaker_config.
    def visitCircuit_breaker_config(self, ctx:apgParser.Circuit_breaker_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cohort_analysis_config.
    def visitCohort_analysis_config(self, ctx:apgParser.Cohort_analysis_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#condition_specification.
    def visitCondition_specification(self, ctx:apgParser.Condition_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#constraint_specification.
    def visitConstraint_specification(self, ctx:apgParser.Constraint_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#delivery_specification.
    def visitDelivery_specification(self, ctx:apgParser.Delivery_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#document_verification_config.
    def visitDocument_verification_config(self, ctx:apgParser.Document_verification_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#escalation_specification.
    def visitEscalation_specification(self, ctx:apgParser.Escalation_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_persistence_config.
    def visitEvent_persistence_config(self, ctx:apgParser.Event_persistence_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fraud_detection_config.
    def visitFraud_detection_config(self, ctx:apgParser.Fraud_detection_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#funnel_analysis_config.
    def visitFunnel_analysis_config(self, ctx:apgParser.Funnel_analysis_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#group_reference.
    def visitGroup_reference(self, ctx:apgParser.Group_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#identity_verification_config.
    def visitIdentity_verification_config(self, ctx:apgParser.Identity_verification_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#kyc_config.
    def visitKyc_config(self, ctx:apgParser.Kyc_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#load_balancer_config.
    def visitLoad_balancer_config(self, ctx:apgParser.Load_balancer_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#localization_specification.
    def visitLocalization_specification(self, ctx:apgParser.Localization_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#monitoring_specification.
    def visitMonitoring_specification(self, ctx:apgParser.Monitoring_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#personalization_specification.
    def visitPersonalization_specification(self, ctx:apgParser.Personalization_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#push_notification_config.
    def visitPush_notification_config(self, ctx:apgParser.Push_notification_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rate_limit_specification.
    def visitRate_limit_specification(self, ctx:apgParser.Rate_limit_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#retry_policy_config.
    def visitRetry_policy_config(self, ctx:apgParser.Retry_policy_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#retry_specification.
    def visitRetry_specification(self, ctx:apgParser.Retry_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#role_reference.
    def visitRole_reference(self, ctx:apgParser.Role_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schedule_specification.
    def visitSchedule_specification(self, ctx:apgParser.Schedule_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schema_specification.
    def visitSchema_specification(self, ctx:apgParser.Schema_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_mesh_config.
    def visitService_mesh_config(self, ctx:apgParser.Service_mesh_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sms_config.
    def visitSms_config(self, ctx:apgParser.Sms_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#template_specification.
    def visitTemplate_specification(self, ctx:apgParser.Template_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#tracking_specification.
    def visitTracking_specification(self, ctx:apgParser.Tracking_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transformation_specification.
    def visitTransformation_specification(self, ctx:apgParser.Transformation_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_reference.
    def visitUser_reference(self, ctx:apgParser.User_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_segmentation_config.
    def visitUser_segmentation_config(self, ctx:apgParser.User_segmentation_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#config_property.
    def visitConfig_property(self, ctx:apgParser.Config_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#policy_property.
    def visitPolicy_property(self, ctx:apgParser.Policy_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#agg_property.
    def visitAgg_property(self, ctx:apgParser.Agg_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#algo_property.
    def visitAlgo_property(self, ctx:apgParser.Algo_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gateway_property.
    def visitGateway_property(self, ctx:apgParser.Gateway_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#api_property.
    def visitApi_property(self, ctx:apgParser.Api_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#resolution_property.
    def visitResolution_property(self, ctx:apgParser.Resolution_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#check_property.
    def visitCheck_property(self, ctx:apgParser.Check_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#backup_property.
    def visitBackup_property(self, ctx:apgParser.Backup_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bi_property.
    def visitBi_property(self, ctx:apgParser.Bi_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#bucket_property.
    def visitBucket_property(self, ctx:apgParser.Bucket_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#chat_property.
    def visitChat_property(self, ctx:apgParser.Chat_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#breaker_property.
    def visitBreaker_property(self, ctx:apgParser.Breaker_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cohort_property.
    def visitCohort_property(self, ctx:apgParser.Cohort_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#constraint_property.
    def visitConstraint_property(self, ctx:apgParser.Constraint_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#delivery_property.
    def visitDelivery_property(self, ctx:apgParser.Delivery_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#doc_property.
    def visitDoc_property(self, ctx:apgParser.Doc_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#escalation_property.
    def visitEscalation_property(self, ctx:apgParser.Escalation_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#persistence_property.
    def visitPersistence_property(self, ctx:apgParser.Persistence_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fraud_property.
    def visitFraud_property(self, ctx:apgParser.Fraud_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#funnel_property.
    def visitFunnel_property(self, ctx:apgParser.Funnel_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#identity_property.
    def visitIdentity_property(self, ctx:apgParser.Identity_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#kyc_property.
    def visitKyc_property(self, ctx:apgParser.Kyc_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#lb_property.
    def visitLb_property(self, ctx:apgParser.Lb_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#localization_property.
    def visitLocalization_property(self, ctx:apgParser.Localization_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#monitoring_property.
    def visitMonitoring_property(self, ctx:apgParser.Monitoring_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#personalization_property.
    def visitPersonalization_property(self, ctx:apgParser.Personalization_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#push_property.
    def visitPush_property(self, ctx:apgParser.Push_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rate_limit_property.
    def visitRate_limit_property(self, ctx:apgParser.Rate_limit_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#retry_property.
    def visitRetry_property(self, ctx:apgParser.Retry_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schedule_property.
    def visitSchedule_property(self, ctx:apgParser.Schedule_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#template_property.
    def visitTemplate_property(self, ctx:apgParser.Template_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#tracking_property.
    def visitTracking_property(self, ctx:apgParser.Tracking_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transform_property.
    def visitTransform_property(self, ctx:apgParser.Transform_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#segment_property.
    def visitSegment_property(self, ctx:apgParser.Segment_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mesh_property.
    def visitMesh_property(self, ctx:apgParser.Mesh_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sms_property.
    def visitSms_property(self, ctx:apgParser.Sms_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#commission_config.
    def visitCommission_config(self, ctx:apgParser.Commission_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#compliance_framework_list.
    def visitCompliance_framework_list(self, ctx:apgParser.Compliance_framework_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#composite_condition.
    def visitComposite_condition(self, ctx:apgParser.Composite_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#compression_specification.
    def visitCompression_specification(self, ctx:apgParser.Compression_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#container_configuration.
    def visitContainer_configuration(self, ctx:apgParser.Container_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#conversion_tracking_config.
    def visitConversion_tracking_config(self, ctx:apgParser.Conversion_tracking_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#correlation_specification.
    def visitCorrelation_specification(self, ctx:apgParser.Correlation_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#currency_conversion_config.
    def visitCurrency_conversion_config(self, ctx:apgParser.Currency_conversion_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#currency_list.
    def visitCurrency_list(self, ctx:apgParser.Currency_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_condition.
    def visitCustom_condition(self, ctx:apgParser.Custom_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_format_specification.
    def visitCustom_format_specification(self, ctx:apgParser.Custom_format_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_unit.
    def visitCustom_unit(self, ctx:apgParser.Custom_unitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dashboard_configuration.
    def visitDashboard_configuration(self, ctx:apgParser.Dashboard_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_access_config.
    def visitData_access_config(self, ctx:apgParser.Data_access_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_warehouse_config.
    def visitData_warehouse_config(self, ctx:apgParser.Data_warehouse_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_reference.
    def visitDatabase_reference(self, ctx:apgParser.Database_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#deployment_environment.
    def visitDeployment_environment(self, ctx:apgParser.Deployment_environmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dispute_config.
    def visitDispute_config(self, ctx:apgParser.Dispute_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#duration_clause.
    def visitDuration_clause(self, ctx:apgParser.Duration_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ecommerce_config.
    def visitEcommerce_config(self, ctx:apgParser.Ecommerce_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#environment_variable.
    def visitEnvironment_variable(self, ctx:apgParser.Environment_variableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#error_handling_config.
    def visitError_handling_config(self, ctx:apgParser.Error_handling_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#escrow_config.
    def visitEscrow_config(self, ctx:apgParser.Escrow_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#execution_environment.
    def visitExecution_environment(self, ctx:apgParser.Execution_environmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#experiment_configuration.
    def visitExperiment_configuration(self, ctx:apgParser.Experiment_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#failover_config.
    def visitFailover_config(self, ctx:apgParser.Failover_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#feature_flag_config.
    def visitFeature_flag_config(self, ctx:apgParser.Feature_flag_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fulfillment_config.
    def visitFulfillment_config(self, ctx:apgParser.Fulfillment_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gdpr_config.
    def visitGdpr_config(self, ctx:apgParser.Gdpr_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#health_check_config.
    def visitHealth_check_config(self, ctx:apgParser.Health_check_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#identity_provider_config.
    def visitIdentity_provider_config(self, ctx:apgParser.Identity_provider_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#infrastructure_requirement.
    def visitInfrastructure_requirement(self, ctx:apgParser.Infrastructure_requirementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inventory_config.
    def visitInventory_config(self, ctx:apgParser.Inventory_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#lambda_configuration.
    def visitLambda_configuration(self, ctx:apgParser.Lambda_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#listing_config.
    def visitListing_config(self, ctx:apgParser.Listing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#machine_learning_config.
    def visitMachine_learning_config(self, ctx:apgParser.Machine_learning_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ml_model_configuration.
    def visitMl_model_configuration(self, ctx:apgParser.Ml_model_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#model_configuration.
    def visitModel_configuration(self, ctx:apgParser.Model_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#notification_template.
    def visitNotification_template(self, ctx:apgParser.Notification_templateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#orchestration_config.
    def visitOrchestration_config(self, ctx:apgParser.Orchestration_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#payment_config.
    def visitPayment_config(self, ctx:apgParser.Payment_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#platform_config.
    def visitPlatform_config(self, ctx:apgParser.Platform_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#platform_name.
    def visitPlatform_name(self, ctx:apgParser.Platform_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#prediction_configuration.
    def visitPrediction_configuration(self, ctx:apgParser.Prediction_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#pricing_config.
    def visitPricing_config(self, ctx:apgParser.Pricing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#quality_gate_config.
    def visitQuality_gate_config(self, ctx:apgParser.Quality_gate_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#region_list.
    def visitRegion_list(self, ctx:apgParser.Region_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#resource_requirement.
    def visitResource_requirement(self, ctx:apgParser.Resource_requirementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#retention_analysis_config.
    def visitRetention_analysis_config(self, ctx:apgParser.Retention_analysis_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#revenue_sharing_config.
    def visitRevenue_sharing_config(self, ctx:apgParser.Revenue_sharing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rollback_config.
    def visitRollback_config(self, ctx:apgParser.Rollback_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#scaling_config.
    def visitScaling_config(self, ctx:apgParser.Scaling_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#security_policy.
    def visitSecurity_policy(self, ctx:apgParser.Security_policyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#subscription_config.
    def visitSubscription_config(self, ctx:apgParser.Subscription_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#tax_calculation_config.
    def visitTax_calculation_config(self, ctx:apgParser.Tax_calculation_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#time_range.
    def visitTime_range(self, ctx:apgParser.Time_rangeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_type_definition.
    def visitUser_type_definition(self, ctx:apgParser.User_type_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#webhook_config.
    def visitWebhook_config(self, ctx:apgParser.Webhook_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ecommerce_name.
    def visitEcommerce_name(self, ctx:apgParser.Ecommerce_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#encryption_specification.
    def visitEncryption_specification(self, ctx:apgParser.Encryption_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#endpoint_definition_list.
    def visitEndpoint_definition_list(self, ctx:apgParser.Endpoint_definition_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enrichment_specification.
    def visitEnrichment_specification(self, ctx:apgParser.Enrichment_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#escalation_policy_specification.
    def visitEscalation_policy_specification(self, ctx:apgParser.Escalation_policy_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_name.
    def visitEvent_name(self, ctx:apgParser.Event_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_routing_config.
    def visitEvent_routing_config(self, ctx:apgParser.Event_routing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_schema.
    def visitEvent_schema(self, ctx:apgParser.Event_schemaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#export_specification.
    def visitExport_specification(self, ctx:apgParser.Export_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#facet_configuration.
    def visitFacet_configuration(self, ctx:apgParser.Facet_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fee_configuration.
    def visitFee_configuration(self, ctx:apgParser.Fee_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#filter_specification.
    def visitFilter_specification(self, ctx:apgParser.Filter_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fraud_prevention_config.
    def visitFraud_prevention_config(self, ctx:apgParser.Fraud_prevention_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#geolocation_config.
    def visitGeolocation_config(self, ctx:apgParser.Geolocation_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#handler_list.
    def visitHandler_list(self, ctx:apgParser.Handler_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#indexing_configuration.
    def visitIndexing_configuration(self, ctx:apgParser.Indexing_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#inventory_management.
    def visitInventory_management(self, ctx:apgParser.Inventory_managementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#label_specification.
    def visitLabel_specification(self, ctx:apgParser.Label_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#load_balancing_config.
    def visitLoad_balancing_config(self, ctx:apgParser.Load_balancing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#localization_config.
    def visitLocalization_config(self, ctx:apgParser.Localization_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#marketplace_name.
    def visitMarketplace_name(self, ctx:apgParser.Marketplace_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#masking_specification.
    def visitMasking_specification(self, ctx:apgParser.Masking_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#messaging_configuration.
    def visitMessaging_configuration(self, ctx:apgParser.Messaging_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#metadata_specification.
    def visitMetadata_specification(self, ctx:apgParser.Metadata_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#metric_reference.
    def visitMetric_reference(self, ctx:apgParser.Metric_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#ml_analytics_config.
    def visitMl_analytics_config(self, ctx:apgParser.Ml_analytics_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#moderation_config.
    def visitModeration_config(self, ctx:apgParser.Moderation_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#monitoring_configuration.
    def visitMonitoring_configuration(self, ctx:apgParser.Monitoring_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#negotiation_config.
    def visitNegotiation_config(self, ctx:apgParser.Negotiation_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#networking_configuration.
    def visitNetworking_configuration(self, ctx:apgParser.Networking_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#onboarding_definition.
    def visitOnboarding_definition(self, ctx:apgParser.Onboarding_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#order_fulfillment.
    def visitOrder_fulfillment(self, ctx:apgParser.Order_fulfillmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#output_specification.
    def visitOutput_specification(self, ctx:apgParser.Output_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#pattern_condition.
    def visitPattern_condition(self, ctx:apgParser.Pattern_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#payment_method_list.
    def visitPayment_method_list(self, ctx:apgParser.Payment_method_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#payment_provider_list.
    def visitPayment_provider_list(self, ctx:apgParser.Payment_provider_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#percentage_clause.
    def visitPercentage_clause(self, ctx:apgParser.Percentage_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#permission_list.
    def visitPermission_list(self, ctx:apgParser.Permission_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#personalization_config.
    def visitPersonalization_config(self, ctx:apgParser.Personalization_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#placement_strategy_config.
    def visitPlacement_strategy_config(self, ctx:apgParser.Placement_strategy_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#predictive_analytics_config.
    def visitPredictive_analytics_config(self, ctx:apgParser.Predictive_analytics_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#quantile_specification.
    def visitQuantile_specification(self, ctx:apgParser.Quantile_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#query_expression.
    def visitQuery_expression(self, ctx:apgParser.Query_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rating_configuration.
    def visitRating_configuration(self, ctx:apgParser.Rating_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#real_time_analytics_config.
    def visitReal_time_analytics_config(self, ctx:apgParser.Real_time_analytics_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#recommendation_config.
    def visitRecommendation_config(self, ctx:apgParser.Recommendation_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#refund_configuration.
    def visitRefund_configuration(self, ctx:apgParser.Refund_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#resource_requirements_config.
    def visitResource_requirements_config(self, ctx:apgParser.Resource_requirements_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#responsibility_list.
    def visitResponsibility_list(self, ctx:apgParser.Responsibility_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#retention_specification.
    def visitRetention_specification(self, ctx:apgParser.Retention_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#review_configuration.
    def visitReview_configuration(self, ctx:apgParser.Review_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rotation_specification.
    def visitRotation_specification(self, ctx:apgParser.Rotation_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#runbook_specification.
    def visitRunbook_specification(self, ctx:apgParser.Runbook_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sampling_specification.
    def visitSampling_specification(self, ctx:apgParser.Sampling_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#scaling_configuration.
    def visitScaling_configuration(self, ctx:apgParser.Scaling_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#search_analytics_config.
    def visitSearch_analytics_config(self, ctx:apgParser.Search_analytics_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#search_engine_type.
    def visitSearch_engine_type(self, ctx:apgParser.Search_engine_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#secrets_config.
    def visitSecrets_config(self, ctx:apgParser.Secrets_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#security_configuration.
    def visitSecurity_configuration(self, ctx:apgParser.Security_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_definition_list.
    def visitService_definition_list(self, ctx:apgParser.Service_definition_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_dependency_list.
    def visitService_dependency_list(self, ctx:apgParser.Service_dependency_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_discovery_config.
    def visitService_discovery_config(self, ctx:apgParser.Service_discovery_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_name.
    def visitService_name(self, ctx:apgParser.Service_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_type.
    def visitService_type(self, ctx:apgParser.Service_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#shipping_zones_config.
    def visitShipping_zones_config(self, ctx:apgParser.Shipping_zones_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#split_configuration.
    def visitSplit_configuration(self, ctx:apgParser.Split_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#storage_configuration.
    def visitStorage_configuration(self, ctx:apgParser.Storage_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#suppression_rules.
    def visitSuppression_rules(self, ctx:apgParser.Suppression_rulesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#threshold_value.
    def visitThreshold_value(self, ctx:apgParser.Threshold_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#tracing_config.
    def visitTracing_config(self, ctx:apgParser.Tracing_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_list.
    def visitTrigger_list(self, ctx:apgParser.Trigger_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_type_name.
    def visitUser_type_name(self, ctx:apgParser.User_type_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#verification_config.
    def visitVerification_config(self, ctx:apgParser.Verification_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#verification_requirements.
    def visitVerification_requirements(self, ctx:apgParser.Verification_requirementsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#video_call_configuration.
    def visitVideo_call_configuration(self, ctx:apgParser.Video_call_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#form_layout.
    def visitForm_layout(self, ctx:apgParser.Form_layoutContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout_type.
    def visitLayout_type(self, ctx:apgParser.Layout_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout_definition.
    def visitLayout_definition(self, ctx:apgParser.Layout_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout_element.
    def visitLayout_element(self, ctx:apgParser.Layout_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#container_element.
    def visitContainer_element(self, ctx:apgParser.Container_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#container_type.
    def visitContainer_type(self, ctx:apgParser.Container_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#field_element.
    def visitField_element(self, ctx:apgParser.Field_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#field_type.
    def visitField_type(self, ctx:apgParser.Field_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#component_element.
    def visitComponent_element(self, ctx:apgParser.Component_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#component_type.
    def visitComponent_type(self, ctx:apgParser.Component_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout_directive.
    def visitLayout_directive(self, ctx:apgParser.Layout_directiveContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#directive_name.
    def visitDirective_name(self, ctx:apgParser.Directive_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#directive_params.
    def visitDirective_params(self, ctx:apgParser.Directive_paramsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#directive_param.
    def visitDirective_param(self, ctx:apgParser.Directive_paramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout_properties.
    def visitLayout_properties(self, ctx:apgParser.Layout_propertiesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#layout_property.
    def visitLayout_property(self, ctx:apgParser.Layout_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#style_property.
    def visitStyle_property(self, ctx:apgParser.Style_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#field_properties.
    def visitField_properties(self, ctx:apgParser.Field_propertiesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#field_property.
    def visitField_property(self, ctx:apgParser.Field_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#component_properties.
    def visitComponent_properties(self, ctx:apgParser.Component_propertiesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#component_property.
    def visitComponent_property(self, ctx:apgParser.Component_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#size_value.
    def visitSize_value(self, ctx:apgParser.Size_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#spacing_value.
    def visitSpacing_value(self, ctx:apgParser.Spacing_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#color_value.
    def visitColor_value(self, ctx:apgParser.Color_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#border_value.
    def visitBorder_value(self, ctx:apgParser.Border_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_value.
    def visitFont_value(self, ctx:apgParser.Font_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#display_value.
    def visitDisplay_value(self, ctx:apgParser.Display_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#position_value.
    def visitPosition_value(self, ctx:apgParser.Position_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#animation_value.
    def visitAnimation_value(self, ctx:apgParser.Animation_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transition_value.
    def visitTransition_value(self, ctx:apgParser.Transition_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#responsive_breakpoints.
    def visitResponsive_breakpoints(self, ctx:apgParser.Responsive_breakpointsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#media_query.
    def visitMedia_query(self, ctx:apgParser.Media_queryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#media_type.
    def visitMedia_type(self, ctx:apgParser.Media_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#media_features.
    def visitMedia_features(self, ctx:apgParser.Media_featuresContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#media_feature.
    def visitMedia_feature(self, ctx:apgParser.Media_featureContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#validation_rules.
    def visitValidation_rules(self, ctx:apgParser.Validation_rulesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#validation_rule.
    def visitValidation_rule(self, ctx:apgParser.Validation_ruleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cross_field_validation.
    def visitCross_field_validation(self, ctx:apgParser.Cross_field_validationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#field_comparison.
    def visitField_comparison(self, ctx:apgParser.Field_comparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#comparison_operator.
    def visitComparison_operator(self, ctx:apgParser.Comparison_operatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#behavior_property.
    def visitBehavior_property(self, ctx:apgParser.Behavior_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_property.
    def visitData_property(self, ctx:apgParser.Data_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#accessibility_property.
    def visitAccessibility_property(self, ctx:apgParser.Accessibility_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transform_value.
    def visitTransform_value(self, ctx:apgParser.Transform_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transform_function.
    def visitTransform_function(self, ctx:apgParser.Transform_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#filter_value.
    def visitFilter_value(self, ctx:apgParser.Filter_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#filter_function.
    def visitFilter_function(self, ctx:apgParser.Filter_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#shadow_value.
    def visitShadow_value(self, ctx:apgParser.Shadow_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gradient_value.
    def visitGradient_value(self, ctx:apgParser.Gradient_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gradient_direction.
    def visitGradient_direction(self, ctx:apgParser.Gradient_directionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gradient_stops.
    def visitGradient_stops(self, ctx:apgParser.Gradient_stopsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gradient_stop.
    def visitGradient_stop(self, ctx:apgParser.Gradient_stopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#clip_value.
    def visitClip_value(self, ctx:apgParser.Clip_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#clip_points.
    def visitClip_points(self, ctx:apgParser.Clip_pointsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#clip_point.
    def visitClip_point(self, ctx:apgParser.Clip_pointContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_value.
    def visitMask_value(self, ctx:apgParser.Mask_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#format_specification.
    def visitFormat_specification(self, ctx:apgParser.Format_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#input_mask.
    def visitInput_mask(self, ctx:apgParser.Input_maskContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_pattern.
    def visitMask_pattern(self, ctx:apgParser.Mask_patternContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_options.
    def visitMask_options(self, ctx:apgParser.Mask_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#autocomplete_spec.
    def visitAutocomplete_spec(self, ctx:apgParser.Autocomplete_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#autocomplete_source.
    def visitAutocomplete_source(self, ctx:apgParser.Autocomplete_sourceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#autocomplete_options.
    def visitAutocomplete_options(self, ctx:apgParser.Autocomplete_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dependency_spec.
    def visitDependency_spec(self, ctx:apgParser.Dependency_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#conditional_spec.
    def visitConditional_spec(self, ctx:apgParser.Conditional_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#accessibility_spec.
    def visitAccessibility_spec(self, ctx:apgParser.Accessibility_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#accessibility_rule.
    def visitAccessibility_rule(self, ctx:apgParser.Accessibility_ruleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#angle_value.
    def visitAngle_value(self, ctx:apgParser.Angle_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#percentage_value.
    def visitPercentage_value(self, ctx:apgParser.Percentage_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#viewport_value.
    def visitViewport_value(self, ctx:apgParser.Viewport_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#calc_expression.
    def visitCalc_expression(self, ctx:apgParser.Calc_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#calc_operand.
    def visitCalc_operand(self, ctx:apgParser.Calc_operandContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#calc_operator.
    def visitCalc_operator(self, ctx:apgParser.Calc_operatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#icon_specification.
    def visitIcon_specification(self, ctx:apgParser.Icon_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#icon_name.
    def visitIcon_name(self, ctx:apgParser.Icon_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#icon_style.
    def visitIcon_style(self, ctx:apgParser.Icon_styleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#icon_size.
    def visitIcon_size(self, ctx:apgParser.Icon_sizeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#action_specification.
    def visitAction_specification(self, ctx:apgParser.Action_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#function_call.
    def visitFunction_call(self, ctx:apgParser.Function_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_handler.
    def visitEvent_handler(self, ctx:apgParser.Event_handlerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#route_action.
    def visitRoute_action(self, ctx:apgParser.Route_actionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_action.
    def visitCustom_action(self, ctx:apgParser.Custom_actionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#action_property.
    def visitAction_property(self, ctx:apgParser.Action_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_specification.
    def visitData_specification(self, ctx:apgParser.Data_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_binding.
    def visitData_binding(self, ctx:apgParser.Data_bindingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_source.
    def visitData_source(self, ctx:apgParser.Data_sourceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_model.
    def visitData_model(self, ctx:apgParser.Data_modelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#state_specification.
    def visitState_specification(self, ctx:apgParser.State_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#state_property.
    def visitState_property(self, ctx:apgParser.State_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_specification.
    def visitTheme_specification(self, ctx:apgParser.Theme_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_definition.
    def visitTheme_definition(self, ctx:apgParser.Theme_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#theme_property.
    def visitTheme_property(self, ctx:apgParser.Theme_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#variant_specification.
    def visitVariant_specification(self, ctx:apgParser.Variant_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#size_specification.
    def visitSize_specification(self, ctx:apgParser.Size_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alignment_specification.
    def visitAlignment_specification(self, ctx:apgParser.Alignment_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#unit.
    def visitUnit(self, ctx:apgParser.UnitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#hex_color.
    def visitHex_color(self, ctx:apgParser.Hex_colorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#rgb_color.
    def visitRgb_color(self, ctx:apgParser.Rgb_colorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#hsl_color.
    def visitHsl_color(self, ctx:apgParser.Hsl_colorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#named_color.
    def visitNamed_color(self, ctx:apgParser.Named_colorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#css_variable.
    def visitCss_variable(self, ctx:apgParser.Css_variableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#border_width.
    def visitBorder_width(self, ctx:apgParser.Border_widthContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#border_style.
    def visitBorder_style(self, ctx:apgParser.Border_styleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#border_color.
    def visitBorder_color(self, ctx:apgParser.Border_colorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_family.
    def visitFont_family(self, ctx:apgParser.Font_familyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_family_name.
    def visitFont_family_name(self, ctx:apgParser.Font_family_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_size.
    def visitFont_size(self, ctx:apgParser.Font_sizeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_size_keyword.
    def visitFont_size_keyword(self, ctx:apgParser.Font_size_keywordContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_weight.
    def visitFont_weight(self, ctx:apgParser.Font_weightContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_weight_keyword.
    def visitFont_weight_keyword(self, ctx:apgParser.Font_weight_keywordContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_style.
    def visitFont_style(self, ctx:apgParser.Font_styleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#animation_name.
    def visitAnimation_name(self, ctx:apgParser.Animation_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#duration.
    def visitDuration(self, ctx:apgParser.DurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#timing_function.
    def visitTiming_function(self, ctx:apgParser.Timing_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#delay.
    def visitDelay(self, ctx:apgParser.DelayContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#iteration_count.
    def visitIteration_count(self, ctx:apgParser.Iteration_countContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#direction.
    def visitDirection(self, ctx:apgParser.DirectionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fill_mode.
    def visitFill_mode(self, ctx:apgParser.Fill_modeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transition_property.
    def visitTransition_property(self, ctx:apgParser.Transition_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#resolution_value.
    def visitResolution_value(self, ctx:apgParser.Resolution_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_validation_function.
    def visitCustom_validation_function(self, ctx:apgParser.Custom_validation_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_binding_spec.
    def visitData_binding_spec(self, ctx:apgParser.Data_binding_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#simple_binding.
    def visitSimple_binding(self, ctx:apgParser.Simple_bindingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#complex_binding.
    def visitComplex_binding(self, ctx:apgParser.Complex_bindingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#binding_property.
    def visitBinding_property(self, ctx:apgParser.Binding_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_source_spec.
    def visitData_source_spec(self, ctx:apgParser.Data_source_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#static_data.
    def visitStatic_data(self, ctx:apgParser.Static_dataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#computed_data.
    def visitComputed_data(self, ctx:apgParser.Computed_dataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#computed_expression.
    def visitComputed_expression(self, ctx:apgParser.Computed_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#lambda_expression.
    def visitLambda_expression(self, ctx:apgParser.Lambda_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#watcher_spec.
    def visitWatcher_spec(self, ctx:apgParser.Watcher_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#watch_property.
    def visitWatch_property(self, ctx:apgParser.Watch_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cache_spec.
    def visitCache_spec(self, ctx:apgParser.Cache_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cache_property.
    def visitCache_property(self, ctx:apgParser.Cache_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gradient_shape.
    def visitGradient_shape(self, ctx:apgParser.Gradient_shapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#gradient_angle.
    def visitGradient_angle(self, ctx:apgParser.Gradient_angleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_source.
    def visitMask_source(self, ctx:apgParser.Mask_sourceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_position.
    def visitMask_position(self, ctx:apgParser.Mask_positionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_size.
    def visitMask_size(self, ctx:apgParser.Mask_sizeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_repeat.
    def visitMask_repeat(self, ctx:apgParser.Mask_repeatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_origin.
    def visitMask_origin(self, ctx:apgParser.Mask_originContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mask_clip.
    def visitMask_clip(self, ctx:apgParser.Mask_clipContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#date_format.
    def visitDate_format(self, ctx:apgParser.Date_formatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#number_format.
    def visitNumber_format(self, ctx:apgParser.Number_formatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#currency_format.
    def visitCurrency_format(self, ctx:apgParser.Currency_formatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_format.
    def visitCustom_format(self, ctx:apgParser.Custom_formatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#color_palette.
    def visitColor_palette(self, ctx:apgParser.Color_paletteContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#color_definition.
    def visitColor_definition(self, ctx:apgParser.Color_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#typography_scale.
    def visitTypography_scale(self, ctx:apgParser.Typography_scaleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#typography_definition.
    def visitTypography_definition(self, ctx:apgParser.Typography_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_definition.
    def visitFont_definition(self, ctx:apgParser.Font_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#font_property.
    def visitFont_property(self, ctx:apgParser.Font_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#spacing_scale.
    def visitSpacing_scale(self, ctx:apgParser.Spacing_scaleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#spacing_definition.
    def visitSpacing_definition(self, ctx:apgParser.Spacing_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#shadow_scale.
    def visitShadow_scale(self, ctx:apgParser.Shadow_scaleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#shadow_definition.
    def visitShadow_definition(self, ctx:apgParser.Shadow_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#api_endpoint.
    def visitApi_endpoint(self, ctx:apgParser.Api_endpointContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#url_string.
    def visitUrl_string(self, ctx:apgParser.Url_stringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#endpoint_config.
    def visitEndpoint_config(self, ctx:apgParser.Endpoint_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#endpoint_property.
    def visitEndpoint_property(self, ctx:apgParser.Endpoint_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#header_property.
    def visitHeader_property(self, ctx:apgParser.Header_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#param_property.
    def visitParam_property(self, ctx:apgParser.Param_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#auth_config.
    def visitAuth_config(self, ctx:apgParser.Auth_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schema_definition.
    def visitSchema_definition(self, ctx:apgParser.Schema_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schema_property.
    def visitSchema_property(self, ctx:apgParser.Schema_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cascade_rule.
    def visitCascade_rule(self, ctx:apgParser.Cascade_ruleContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#cascade_step.
    def visitCascade_step(self, ctx:apgParser.Cascade_stepContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#condition_expression.
    def visitCondition_expression(self, ctx:apgParser.Condition_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#object_property.
    def visitObject_property(self, ctx:apgParser.Object_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#parameter_list.
    def visitParameter_list(self, ctx:apgParser.Parameter_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#duration_value.
    def visitDuration_value(self, ctx:apgParser.Duration_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#time_unit.
    def visitTime_unit(self, ctx:apgParser.Time_unitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_definition.
    def visitTest_definition(self, ctx:apgParser.Test_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_type.
    def visitTest_type(self, ctx:apgParser.Test_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_configuration.
    def visitTest_configuration(self, ctx:apgParser.Test_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_config_item.
    def visitTest_config_item(self, ctx:apgParser.Test_config_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#browser_specification.
    def visitBrowser_specification(self, ctx:apgParser.Browser_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#browser_config.
    def visitBrowser_config(self, ctx:apgParser.Browser_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#browser_property.
    def visitBrowser_property(self, ctx:apgParser.Browser_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#device_specification.
    def visitDevice_specification(self, ctx:apgParser.Device_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#device_config.
    def visitDevice_config(self, ctx:apgParser.Device_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#device_property.
    def visitDevice_property(self, ctx:apgParser.Device_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_body.
    def visitTest_body(self, ctx:apgParser.Test_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_step.
    def visitTest_step(self, ctx:apgParser.Test_stepContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_action.
    def visitTest_action(self, ctx:apgParser.Test_actionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_setup.
    def visitTest_setup(self, ctx:apgParser.Test_setupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#setup_action.
    def visitSetup_action(self, ctx:apgParser.Setup_actionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_verification.
    def visitTest_verification(self, ctx:apgParser.Test_verificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#verification_condition.
    def visitVerification_condition(self, ctx:apgParser.Verification_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#element_verification.
    def visitElement_verification(self, ctx:apgParser.Element_verificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#state_verification.
    def visitState_verification(self, ctx:apgParser.State_verificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#data_verification.
    def visitData_verification(self, ctx:apgParser.Data_verificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#test_assertion.
    def visitTest_assertion(self, ctx:apgParser.Test_assertionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#assertion_expression.
    def visitAssertion_expression(self, ctx:apgParser.Assertion_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#expectation_expression.
    def visitExpectation_expression(self, ctx:apgParser.Expectation_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#async_expectation.
    def visitAsync_expectation(self, ctx:apgParser.Async_expectationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#verification_expression.
    def visitVerification_expression(self, ctx:apgParser.Verification_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#element_state_verification.
    def visitElement_state_verification(self, ctx:apgParser.Element_state_verificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#element_state.
    def visitElement_state(self, ctx:apgParser.Element_stateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#element_property.
    def visitElement_property(self, ctx:apgParser.Element_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exception_type.
    def visitException_type(self, ctx:apgParser.Exception_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#selector_expr.
    def visitSelector_expr(self, ctx:apgParser.Selector_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#css_selector.
    def visitCss_selector(self, ctx:apgParser.Css_selectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#xpath_selector.
    def visitXpath_selector(self, ctx:apgParser.Xpath_selectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#id_selector.
    def visitId_selector(self, ctx:apgParser.Id_selectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#class_selector.
    def visitClass_selector(self, ctx:apgParser.Class_selectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#scroll_options.
    def visitScroll_options(self, ctx:apgParser.Scroll_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#scroll_property.
    def visitScroll_property(self, ctx:apgParser.Scroll_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#scroll_direction.
    def visitScroll_direction(self, ctx:apgParser.Scroll_directionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#option_expr.
    def visitOption_expr(self, ctx:apgParser.Option_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#code_block.
    def visitCode_block(self, ctx:apgParser.Code_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#property_matcher.
    def visitProperty_matcher(self, ctx:apgParser.Property_matcherContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#custom_matcher.
    def visitCustom_matcher(self, ctx:apgParser.Custom_matcherContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#spy_options.
    def visitSpy_options(self, ctx:apgParser.Spy_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#spy_property.
    def visitSpy_property(self, ctx:apgParser.Spy_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#stub_options.
    def visitStub_options(self, ctx:apgParser.Stub_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#stub_property.
    def visitStub_property(self, ctx:apgParser.Stub_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fake_options.
    def visitFake_options(self, ctx:apgParser.Fake_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#fake_property.
    def visitFake_property(self, ctx:apgParser.Fake_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trace_expression.
    def visitTrace_expression(self, ctx:apgParser.Trace_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#profile_expression.
    def visitProfile_expression(self, ctx:apgParser.Profile_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#benchmark_expression.
    def visitBenchmark_expression(self, ctx:apgParser.Benchmark_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vault_specification.
    def visitVault_specification(self, ctx:apgParser.Vault_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vault_provider.
    def visitVault_provider(self, ctx:apgParser.Vault_providerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vault_config.
    def visitVault_config(self, ctx:apgParser.Vault_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vault_property.
    def visitVault_property(self, ctx:apgParser.Vault_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_specification.
    def visitDatabase_specification(self, ctx:apgParser.Database_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_type.
    def visitDatabase_type(self, ctx:apgParser.Database_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_config.
    def visitDatabase_config(self, ctx:apgParser.Database_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_property.
    def visitDatabase_property(self, ctx:apgParser.Database_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#database_schema.
    def visitDatabase_schema(self, ctx:apgParser.Database_schemaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#schema_element.
    def visitSchema_element(self, ctx:apgParser.Schema_elementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_definition.
    def visitTable_definition(self, ctx:apgParser.Table_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_alias.
    def visitTable_alias(self, ctx:apgParser.Table_aliasContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_note.
    def visitTable_note(self, ctx:apgParser.Table_noteContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_body.
    def visitTable_body(self, ctx:apgParser.Table_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_definition.
    def visitColumn_definition(self, ctx:apgParser.Column_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_type.
    def visitColumn_type(self, ctx:apgParser.Column_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_data_type.
    def visitDb_data_type(self, ctx:apgParser.Db_data_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_nullable.
    def visitColumn_nullable(self, ctx:apgParser.Column_nullableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_constraints.
    def visitColumn_constraints(self, ctx:apgParser.Column_constraintsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_constraint.
    def visitColumn_constraint(self, ctx:apgParser.Column_constraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_spec.
    def visitReference_spec(self, ctx:apgParser.Reference_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_type.
    def visitReference_type(self, ctx:apgParser.Reference_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_column_ref.
    def visitTable_column_ref(self, ctx:apgParser.Table_column_refContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_reference.
    def visitTable_reference(self, ctx:apgParser.Table_referenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_options.
    def visitReference_options(self, ctx:apgParser.Reference_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_option.
    def visitReference_option(self, ctx:apgParser.Reference_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#reference_action.
    def visitReference_action(self, ctx:apgParser.Reference_actionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_index.
    def visitTable_index(self, ctx:apgParser.Table_indexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_definition.
    def visitIndex_definition(self, ctx:apgParser.Index_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_columns.
    def visitIndex_columns(self, ctx:apgParser.Index_columnsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_options.
    def visitIndex_options(self, ctx:apgParser.Index_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_option.
    def visitIndex_option(self, ctx:apgParser.Index_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#index_type.
    def visitIndex_type(self, ctx:apgParser.Index_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_constraint.
    def visitTable_constraint(self, ctx:apgParser.Table_constraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#constraint_definition.
    def visitConstraint_definition(self, ctx:apgParser.Constraint_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_definition.
    def visitEnum_definition(self, ctx:apgParser.Enum_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_values.
    def visitEnum_values(self, ctx:apgParser.Enum_valuesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_value.
    def visitEnum_value(self, ctx:apgParser.Enum_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#enum_note.
    def visitEnum_note(self, ctx:apgParser.Enum_noteContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_group.
    def visitTable_group(self, ctx:apgParser.Table_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#note_definition.
    def visitNote_definition(self, ctx:apgParser.Note_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_definition.
    def visitTrigger_definition(self, ctx:apgParser.Trigger_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_spec.
    def visitTrigger_spec(self, ctx:apgParser.Trigger_specContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_timing.
    def visitTrigger_timing(self, ctx:apgParser.Trigger_timingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_event.
    def visitTrigger_event(self, ctx:apgParser.Trigger_eventContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_condition.
    def visitTrigger_condition(self, ctx:apgParser.Trigger_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_level.
    def visitTrigger_level(self, ctx:apgParser.Trigger_levelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_body.
    def visitTrigger_body(self, ctx:apgParser.Trigger_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trigger_statement.
    def visitTrigger_statement(self, ctx:apgParser.Trigger_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_definition.
    def visitProcedure_definition(self, ctx:apgParser.Procedure_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_parameters.
    def visitProcedure_parameters(self, ctx:apgParser.Procedure_parametersContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_parameter.
    def visitProcedure_parameter(self, ctx:apgParser.Procedure_parameterContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#parameter_mode.
    def visitParameter_mode(self, ctx:apgParser.Parameter_modeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#default_value.
    def visitDefault_value(self, ctx:apgParser.Default_valueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_options.
    def visitProcedure_options(self, ctx:apgParser.Procedure_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_option.
    def visitProcedure_option(self, ctx:apgParser.Procedure_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_language.
    def visitProcedure_language(self, ctx:apgParser.Procedure_languageContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_body.
    def visitProcedure_body(self, ctx:apgParser.Procedure_bodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#procedure_statement.
    def visitProcedure_statement(self, ctx:apgParser.Procedure_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#function_definition.
    def visitFunction_definition(self, ctx:apgParser.Function_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_return_type.
    def visitDb_return_type(self, ctx:apgParser.Db_return_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_column_list.
    def visitTable_column_list(self, ctx:apgParser.Table_column_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#table_column_def.
    def visitTable_column_def(self, ctx:apgParser.Table_column_defContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#function_options.
    def visitFunction_options(self, ctx:apgParser.Function_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#function_option.
    def visitFunction_option(self, ctx:apgParser.Function_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#view_definition.
    def visitView_definition(self, ctx:apgParser.View_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#view_options.
    def visitView_options(self, ctx:apgParser.View_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#view_option.
    def visitView_option(self, ctx:apgParser.View_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sql_statement.
    def visitSql_statement(self, ctx:apgParser.Sql_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sql_query.
    def visitSql_query(self, ctx:apgParser.Sql_queryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#select_statement.
    def visitSelect_statement(self, ctx:apgParser.Select_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#select_list.
    def visitSelect_list(self, ctx:apgParser.Select_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#select_item.
    def visitSelect_item(self, ctx:apgParser.Select_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#from_clause.
    def visitFrom_clause(self, ctx:apgParser.From_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#where_clause.
    def visitWhere_clause(self, ctx:apgParser.Where_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#group_by_clause.
    def visitGroup_by_clause(self, ctx:apgParser.Group_by_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#having_clause.
    def visitHaving_clause(self, ctx:apgParser.Having_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#order_by_clause.
    def visitOrder_by_clause(self, ctx:apgParser.Order_by_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#order_item.
    def visitOrder_item(self, ctx:apgParser.Order_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#limit_clause.
    def visitLimit_clause(self, ctx:apgParser.Limit_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#insert_statement.
    def visitInsert_statement(self, ctx:apgParser.Insert_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#update_statement.
    def visitUpdate_statement(self, ctx:apgParser.Update_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#delete_statement.
    def visitDelete_statement(self, ctx:apgParser.Delete_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#execute_statement.
    def visitExecute_statement(self, ctx:apgParser.Execute_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#column_list.
    def visitColumn_list(self, ctx:apgParser.Column_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#value_list.
    def visitValue_list(self, ctx:apgParser.Value_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#assignment_list.
    def visitAssignment_list(self, ctx:apgParser.Assignment_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_assignment.
    def visitDb_assignment(self, ctx:apgParser.Db_assignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#control_flow_statement.
    def visitControl_flow_statement(self, ctx:apgParser.Control_flow_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#loop_statement.
    def visitLoop_statement(self, ctx:apgParser.Loop_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#case_statement.
    def visitCase_statement(self, ctx:apgParser.Case_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_when_clause.
    def visitDb_when_clause(self, ctx:apgParser.Db_when_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_else_clause.
    def visitDb_else_clause(self, ctx:apgParser.Db_else_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exception_handling.
    def visitException_handling(self, ctx:apgParser.Exception_handlingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#exception_condition.
    def visitException_condition(self, ctx:apgParser.Exception_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_return_statement.
    def visitDb_return_statement(self, ctx:apgParser.Db_return_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_index_definition.
    def visitVector_index_definition(self, ctx:apgParser.Vector_index_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_index_options.
    def visitVector_index_options(self, ctx:apgParser.Vector_index_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_index_option.
    def visitVector_index_option(self, ctx:apgParser.Vector_index_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_index_method.
    def visitVector_index_method(self, ctx:apgParser.Vector_index_methodContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#distance_function.
    def visitDistance_function(self, ctx:apgParser.Distance_functionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_constraint.
    def visitVector_constraint(self, ctx:apgParser.Vector_constraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_constraint_expression.
    def visitVector_constraint_expression(self, ctx:apgParser.Vector_constraint_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#vector_column_constraint.
    def visitVector_column_constraint(self, ctx:apgParser.Vector_column_constraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#apg_statement.
    def visitApg_statement(self, ctx:apgParser.Apg_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#sql_expression.
    def visitSql_expression(self, ctx:apgParser.Sql_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_parameter_list.
    def visitDb_parameter_list(self, ctx:apgParser.Db_parameter_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#db_statement_block.
    def visitDb_statement_block(self, ctx:apgParser.Db_statement_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#matcher_expression.
    def visitMatcher_expression(self, ctx:apgParser.Matcher_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mock_configuration.
    def visitMock_configuration(self, ctx:apgParser.Mock_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mock_item.
    def visitMock_item(self, ctx:apgParser.Mock_itemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mock_specification.
    def visitMock_specification(self, ctx:apgParser.Mock_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mock_options.
    def visitMock_options(self, ctx:apgParser.Mock_optionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#mock_option.
    def visitMock_option(self, ctx:apgParser.Mock_optionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#debug_statement.
    def visitDebug_statement(self, ctx:apgParser.Debug_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#debug_expression.
    def visitDebug_expression(self, ctx:apgParser.Debug_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#debug_info.
    def visitDebug_info(self, ctx:apgParser.Debug_infoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#breakpoint_condition.
    def visitBreakpoint_condition(self, ctx:apgParser.Breakpoint_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#config_definition.
    def visitConfig_definition(self, ctx:apgParser.Config_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#config_scope.
    def visitConfig_scope(self, ctx:apgParser.Config_scopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#config_source.
    def visitConfig_source(self, ctx:apgParser.Config_sourceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#config_validation.
    def visitConfig_validation(self, ctx:apgParser.Config_validationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#secret_definition.
    def visitSecret_definition(self, ctx:apgParser.Secret_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#notification_definition.
    def visitNotification_definition(self, ctx:apgParser.Notification_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#notification_type.
    def visitNotification_type(self, ctx:apgParser.Notification_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#notification_configuration.
    def visitNotification_configuration(self, ctx:apgParser.Notification_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#notification_property.
    def visitNotification_property(self, ctx:apgParser.Notification_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#recipient_list.
    def visitRecipient_list(self, ctx:apgParser.Recipient_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#recipient.
    def visitRecipient(self, ctx:apgParser.RecipientContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#recipient_group.
    def visitRecipient_group(self, ctx:apgParser.Recipient_groupContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#dynamic_recipient_list.
    def visitDynamic_recipient_list(self, ctx:apgParser.Dynamic_recipient_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#priority_level.
    def visitPriority_level(self, ctx:apgParser.Priority_levelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alert_definition.
    def visitAlert_definition(self, ctx:apgParser.Alert_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alert_type.
    def visitAlert_type(self, ctx:apgParser.Alert_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alert_configuration.
    def visitAlert_configuration(self, ctx:apgParser.Alert_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alert_property.
    def visitAlert_property(self, ctx:apgParser.Alert_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#alert_condition.
    def visitAlert_condition(self, ctx:apgParser.Alert_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#threshold_condition.
    def visitThreshold_condition(self, ctx:apgParser.Threshold_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#anomaly_condition.
    def visitAnomaly_condition(self, ctx:apgParser.Anomaly_conditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#severity_level.
    def visitSeverity_level(self, ctx:apgParser.Severity_levelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#logger_definition.
    def visitLogger_definition(self, ctx:apgParser.Logger_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#logger_type.
    def visitLogger_type(self, ctx:apgParser.Logger_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#logger_configuration.
    def visitLogger_configuration(self, ctx:apgParser.Logger_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#logger_property.
    def visitLogger_property(self, ctx:apgParser.Logger_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#log_level.
    def visitLog_level(self, ctx:apgParser.Log_levelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#log_format.
    def visitLog_format(self, ctx:apgParser.Log_formatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#metric_definition.
    def visitMetric_definition(self, ctx:apgParser.Metric_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#metric_type.
    def visitMetric_type(self, ctx:apgParser.Metric_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#metric_configuration.
    def visitMetric_configuration(self, ctx:apgParser.Metric_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#metric_property.
    def visitMetric_property(self, ctx:apgParser.Metric_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#unit_specification.
    def visitUnit_specification(self, ctx:apgParser.Unit_specificationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#marketplace_entity.
    def visitMarketplace_entity(self, ctx:apgParser.Marketplace_entityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#marketplace_config.
    def visitMarketplace_config(self, ctx:apgParser.Marketplace_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#marketplace_component.
    def visitMarketplace_component(self, ctx:apgParser.Marketplace_componentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_types_definition.
    def visitUser_types_definition(self, ctx:apgParser.User_types_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_type_list.
    def visitUser_type_list(self, ctx:apgParser.User_type_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_type.
    def visitUser_type(self, ctx:apgParser.User_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_type_config.
    def visitUser_type_config(self, ctx:apgParser.User_type_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#user_type_property.
    def visitUser_type_property(self, ctx:apgParser.User_type_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transaction_engine.
    def visitTransaction_engine(self, ctx:apgParser.Transaction_engineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transaction_config.
    def visitTransaction_config(self, ctx:apgParser.Transaction_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#transaction_property.
    def visitTransaction_property(self, ctx:apgParser.Transaction_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trust_safety_system.
    def visitTrust_safety_system(self, ctx:apgParser.Trust_safety_systemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trust_safety_config.
    def visitTrust_safety_config(self, ctx:apgParser.Trust_safety_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#trust_safety_property.
    def visitTrust_safety_property(self, ctx:apgParser.Trust_safety_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#search_discovery_engine.
    def visitSearch_discovery_engine(self, ctx:apgParser.Search_discovery_engineContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#search_config.
    def visitSearch_config(self, ctx:apgParser.Search_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#search_property.
    def visitSearch_property(self, ctx:apgParser.Search_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#communication_system.
    def visitCommunication_system(self, ctx:apgParser.Communication_systemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#communication_config.
    def visitCommunication_config(self, ctx:apgParser.Communication_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#communication_property.
    def visitCommunication_property(self, ctx:apgParser.Communication_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#microservices_architecture.
    def visitMicroservices_architecture(self, ctx:apgParser.Microservices_architectureContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#microservices_config.
    def visitMicroservices_config(self, ctx:apgParser.Microservices_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#microservices_property.
    def visitMicroservices_property(self, ctx:apgParser.Microservices_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_definition.
    def visitService_definition(self, ctx:apgParser.Service_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_config.
    def visitService_config(self, ctx:apgParser.Service_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#service_property.
    def visitService_property(self, ctx:apgParser.Service_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#deployment_configuration.
    def visitDeployment_configuration(self, ctx:apgParser.Deployment_configurationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#deployment_config.
    def visitDeployment_config(self, ctx:apgParser.Deployment_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#deployment_property.
    def visitDeployment_property(self, ctx:apgParser.Deployment_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#internationalization_config.
    def visitInternationalization_config(self, ctx:apgParser.Internationalization_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#i18n_config.
    def visitI18n_config(self, ctx:apgParser.I18n_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#i18n_property.
    def visitI18n_property(self, ctx:apgParser.I18n_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#analytics_intelligence.
    def visitAnalytics_intelligence(self, ctx:apgParser.Analytics_intelligenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#analytics_config.
    def visitAnalytics_config(self, ctx:apgParser.Analytics_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#analytics_property.
    def visitAnalytics_property(self, ctx:apgParser.Analytics_propertyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#marketplace_events.
    def visitMarketplace_events(self, ctx:apgParser.Marketplace_eventsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_definitions.
    def visitEvent_definitions(self, ctx:apgParser.Event_definitionsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_definition.
    def visitEvent_definition(self, ctx:apgParser.Event_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_config.
    def visitEvent_config(self, ctx:apgParser.Event_configContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by apgParser#event_property.
    def visitEvent_property(self, ctx:apgParser.Event_propertyContext):
        return self.visitChildren(ctx)



del apgParser
