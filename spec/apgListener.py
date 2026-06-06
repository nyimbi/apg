# Generated from apg.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .apgParser import apgParser
else:
    from apgParser import apgParser

# This class defines a complete listener for a parse tree produced by apgParser.
class apgListener(ParseTreeListener):

    # Enter a parse tree produced by apgParser#program.
    def enterProgram(self, ctx:apgParser.ProgramContext):
        pass

    # Exit a parse tree produced by apgParser#program.
    def exitProgram(self, ctx:apgParser.ProgramContext):
        pass


    # Enter a parse tree produced by apgParser#module_declaration.
    def enterModule_declaration(self, ctx:apgParser.Module_declarationContext):
        pass

    # Exit a parse tree produced by apgParser#module_declaration.
    def exitModule_declaration(self, ctx:apgParser.Module_declarationContext):
        pass


    # Enter a parse tree produced by apgParser#module_name.
    def enterModule_name(self, ctx:apgParser.Module_nameContext):
        pass

    # Exit a parse tree produced by apgParser#module_name.
    def exitModule_name(self, ctx:apgParser.Module_nameContext):
        pass


    # Enter a parse tree produced by apgParser#module_metadata.
    def enterModule_metadata(self, ctx:apgParser.Module_metadataContext):
        pass

    # Exit a parse tree produced by apgParser#module_metadata.
    def exitModule_metadata(self, ctx:apgParser.Module_metadataContext):
        pass


    # Enter a parse tree produced by apgParser#module_property.
    def enterModule_property(self, ctx:apgParser.Module_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#module_property.
    def exitModule_property(self, ctx:apgParser.Module_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#dependency_list.
    def enterDependency_list(self, ctx:apgParser.Dependency_listContext):
        pass

    # Exit a parse tree produced by apgParser#dependency_list.
    def exitDependency_list(self, ctx:apgParser.Dependency_listContext):
        pass


    # Enter a parse tree produced by apgParser#dependency.
    def enterDependency(self, ctx:apgParser.DependencyContext):
        pass

    # Exit a parse tree produced by apgParser#dependency.
    def exitDependency(self, ctx:apgParser.DependencyContext):
        pass


    # Enter a parse tree produced by apgParser#version_constraint.
    def enterVersion_constraint(self, ctx:apgParser.Version_constraintContext):
        pass

    # Exit a parse tree produced by apgParser#version_constraint.
    def exitVersion_constraint(self, ctx:apgParser.Version_constraintContext):
        pass


    # Enter a parse tree produced by apgParser#version_range.
    def enterVersion_range(self, ctx:apgParser.Version_rangeContext):
        pass

    # Exit a parse tree produced by apgParser#version_range.
    def exitVersion_range(self, ctx:apgParser.Version_rangeContext):
        pass


    # Enter a parse tree produced by apgParser#export_list.
    def enterExport_list(self, ctx:apgParser.Export_listContext):
        pass

    # Exit a parse tree produced by apgParser#export_list.
    def exitExport_list(self, ctx:apgParser.Export_listContext):
        pass


    # Enter a parse tree produced by apgParser#export_item.
    def enterExport_item(self, ctx:apgParser.Export_itemContext):
        pass

    # Exit a parse tree produced by apgParser#export_item.
    def exitExport_item(self, ctx:apgParser.Export_itemContext):
        pass


    # Enter a parse tree produced by apgParser#private_list.
    def enterPrivate_list(self, ctx:apgParser.Private_listContext):
        pass

    # Exit a parse tree produced by apgParser#private_list.
    def exitPrivate_list(self, ctx:apgParser.Private_listContext):
        pass


    # Enter a parse tree produced by apgParser#alias.
    def enterAlias(self, ctx:apgParser.AliasContext):
        pass

    # Exit a parse tree produced by apgParser#alias.
    def exitAlias(self, ctx:apgParser.AliasContext):
        pass


    # Enter a parse tree produced by apgParser#import_statement.
    def enterImport_statement(self, ctx:apgParser.Import_statementContext):
        pass

    # Exit a parse tree produced by apgParser#import_statement.
    def exitImport_statement(self, ctx:apgParser.Import_statementContext):
        pass


    # Enter a parse tree produced by apgParser#include_statement.
    def enterInclude_statement(self, ctx:apgParser.Include_statementContext):
        pass

    # Exit a parse tree produced by apgParser#include_statement.
    def exitInclude_statement(self, ctx:apgParser.Include_statementContext):
        pass


    # Enter a parse tree produced by apgParser#export_statement.
    def enterExport_statement(self, ctx:apgParser.Export_statementContext):
        pass

    # Exit a parse tree produced by apgParser#export_statement.
    def exitExport_statement(self, ctx:apgParser.Export_statementContext):
        pass


    # Enter a parse tree produced by apgParser#export_declaration.
    def enterExport_declaration(self, ctx:apgParser.Export_declarationContext):
        pass

    # Exit a parse tree produced by apgParser#export_declaration.
    def exitExport_declaration(self, ctx:apgParser.Export_declarationContext):
        pass


    # Enter a parse tree produced by apgParser#module_path.
    def enterModule_path(self, ctx:apgParser.Module_pathContext):
        pass

    # Exit a parse tree produced by apgParser#module_path.
    def exitModule_path(self, ctx:apgParser.Module_pathContext):
        pass


    # Enter a parse tree produced by apgParser#file_path.
    def enterFile_path(self, ctx:apgParser.File_pathContext):
        pass

    # Exit a parse tree produced by apgParser#file_path.
    def exitFile_path(self, ctx:apgParser.File_pathContext):
        pass


    # Enter a parse tree produced by apgParser#import_options.
    def enterImport_options(self, ctx:apgParser.Import_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#import_options.
    def exitImport_options(self, ctx:apgParser.Import_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#import_option.
    def enterImport_option(self, ctx:apgParser.Import_optionContext):
        pass

    # Exit a parse tree produced by apgParser#import_option.
    def exitImport_option(self, ctx:apgParser.Import_optionContext):
        pass


    # Enter a parse tree produced by apgParser#include_options.
    def enterInclude_options(self, ctx:apgParser.Include_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#include_options.
    def exitInclude_options(self, ctx:apgParser.Include_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#include_option.
    def enterInclude_option(self, ctx:apgParser.Include_optionContext):
        pass

    # Exit a parse tree produced by apgParser#include_option.
    def exitInclude_option(self, ctx:apgParser.Include_optionContext):
        pass


    # Enter a parse tree produced by apgParser#import_list.
    def enterImport_list(self, ctx:apgParser.Import_listContext):
        pass

    # Exit a parse tree produced by apgParser#import_list.
    def exitImport_list(self, ctx:apgParser.Import_listContext):
        pass


    # Enter a parse tree produced by apgParser#import_item.
    def enterImport_item(self, ctx:apgParser.Import_itemContext):
        pass

    # Exit a parse tree produced by apgParser#import_item.
    def exitImport_item(self, ctx:apgParser.Import_itemContext):
        pass


    # Enter a parse tree produced by apgParser#entity.
    def enterEntity(self, ctx:apgParser.EntityContext):
        pass

    # Exit a parse tree produced by apgParser#entity.
    def exitEntity(self, ctx:apgParser.EntityContext):
        pass


    # Enter a parse tree produced by apgParser#decorator.
    def enterDecorator(self, ctx:apgParser.DecoratorContext):
        pass

    # Exit a parse tree produced by apgParser#decorator.
    def exitDecorator(self, ctx:apgParser.DecoratorContext):
        pass


    # Enter a parse tree produced by apgParser#entity_type.
    def enterEntity_type(self, ctx:apgParser.Entity_typeContext):
        pass

    # Exit a parse tree produced by apgParser#entity_type.
    def exitEntity_type(self, ctx:apgParser.Entity_typeContext):
        pass


    # Enter a parse tree produced by apgParser#inheritance.
    def enterInheritance(self, ctx:apgParser.InheritanceContext):
        pass

    # Exit a parse tree produced by apgParser#inheritance.
    def exitInheritance(self, ctx:apgParser.InheritanceContext):
        pass


    # Enter a parse tree produced by apgParser#version_tag.
    def enterVersion_tag(self, ctx:apgParser.Version_tagContext):
        pass

    # Exit a parse tree produced by apgParser#version_tag.
    def exitVersion_tag(self, ctx:apgParser.Version_tagContext):
        pass


    # Enter a parse tree produced by apgParser#entity_body.
    def enterEntity_body(self, ctx:apgParser.Entity_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#entity_body.
    def exitEntity_body(self, ctx:apgParser.Entity_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#entity_member.
    def enterEntity_member(self, ctx:apgParser.Entity_memberContext):
        pass

    # Exit a parse tree produced by apgParser#entity_member.
    def exitEntity_member(self, ctx:apgParser.Entity_memberContext):
        pass


    # Enter a parse tree produced by apgParser#config_item.
    def enterConfig_item(self, ctx:apgParser.Config_itemContext):
        pass

    # Exit a parse tree produced by apgParser#config_item.
    def exitConfig_item(self, ctx:apgParser.Config_itemContext):
        pass


    # Enter a parse tree produced by apgParser#capability_contract_block.
    def enterCapability_contract_block(self, ctx:apgParser.Capability_contract_blockContext):
        pass

    # Exit a parse tree produced by apgParser#capability_contract_block.
    def exitCapability_contract_block(self, ctx:apgParser.Capability_contract_blockContext):
        pass


    # Enter a parse tree produced by apgParser#capability_contract.
    def enterCapability_contract(self, ctx:apgParser.Capability_contractContext):
        pass

    # Exit a parse tree produced by apgParser#capability_contract.
    def exitCapability_contract(self, ctx:apgParser.Capability_contractContext):
        pass


    # Enter a parse tree produced by apgParser#capability_contract_member.
    def enterCapability_contract_member(self, ctx:apgParser.Capability_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#capability_contract_member.
    def exitCapability_contract_member(self, ctx:apgParser.Capability_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#erp_component_block.
    def enterErp_component_block(self, ctx:apgParser.Erp_component_blockContext):
        pass

    # Exit a parse tree produced by apgParser#erp_component_block.
    def exitErp_component_block(self, ctx:apgParser.Erp_component_blockContext):
        pass


    # Enter a parse tree produced by apgParser#erp_module_set.
    def enterErp_module_set(self, ctx:apgParser.Erp_module_setContext):
        pass

    # Exit a parse tree produced by apgParser#erp_module_set.
    def exitErp_module_set(self, ctx:apgParser.Erp_module_setContext):
        pass


    # Enter a parse tree produced by apgParser#erp_component_set.
    def enterErp_component_set(self, ctx:apgParser.Erp_component_setContext):
        pass

    # Exit a parse tree produced by apgParser#erp_component_set.
    def exitErp_component_set(self, ctx:apgParser.Erp_component_setContext):
        pass


    # Enter a parse tree produced by apgParser#erp_component_binding.
    def enterErp_component_binding(self, ctx:apgParser.Erp_component_bindingContext):
        pass

    # Exit a parse tree produced by apgParser#erp_component_binding.
    def exitErp_component_binding(self, ctx:apgParser.Erp_component_bindingContext):
        pass


    # Enter a parse tree produced by apgParser#erp_component_key.
    def enterErp_component_key(self, ctx:apgParser.Erp_component_keyContext):
        pass

    # Exit a parse tree produced by apgParser#erp_component_key.
    def exitErp_component_key(self, ctx:apgParser.Erp_component_keyContext):
        pass


    # Enter a parse tree produced by apgParser#erp_component_ref.
    def enterErp_component_ref(self, ctx:apgParser.Erp_component_refContext):
        pass

    # Exit a parse tree produced by apgParser#erp_component_ref.
    def exitErp_component_ref(self, ctx:apgParser.Erp_component_refContext):
        pass


    # Enter a parse tree produced by apgParser#erp_domain.
    def enterErp_domain(self, ctx:apgParser.Erp_domainContext):
        pass

    # Exit a parse tree produced by apgParser#erp_domain.
    def exitErp_domain(self, ctx:apgParser.Erp_domainContext):
        pass


    # Enter a parse tree produced by apgParser#erp_component_member.
    def enterErp_component_member(self, ctx:apgParser.Erp_component_memberContext):
        pass

    # Exit a parse tree produced by apgParser#erp_component_member.
    def exitErp_component_member(self, ctx:apgParser.Erp_component_memberContext):
        pass


    # Enter a parse tree produced by apgParser#erp_data_contract.
    def enterErp_data_contract(self, ctx:apgParser.Erp_data_contractContext):
        pass

    # Exit a parse tree produced by apgParser#erp_data_contract.
    def exitErp_data_contract(self, ctx:apgParser.Erp_data_contractContext):
        pass


    # Enter a parse tree produced by apgParser#erp_data_member.
    def enterErp_data_member(self, ctx:apgParser.Erp_data_memberContext):
        pass

    # Exit a parse tree produced by apgParser#erp_data_member.
    def exitErp_data_member(self, ctx:apgParser.Erp_data_memberContext):
        pass


    # Enter a parse tree produced by apgParser#erp_api_contract.
    def enterErp_api_contract(self, ctx:apgParser.Erp_api_contractContext):
        pass

    # Exit a parse tree produced by apgParser#erp_api_contract.
    def exitErp_api_contract(self, ctx:apgParser.Erp_api_contractContext):
        pass


    # Enter a parse tree produced by apgParser#erp_api_member.
    def enterErp_api_member(self, ctx:apgParser.Erp_api_memberContext):
        pass

    # Exit a parse tree produced by apgParser#erp_api_member.
    def exitErp_api_member(self, ctx:apgParser.Erp_api_memberContext):
        pass


    # Enter a parse tree produced by apgParser#erp_workflow_contract.
    def enterErp_workflow_contract(self, ctx:apgParser.Erp_workflow_contractContext):
        pass

    # Exit a parse tree produced by apgParser#erp_workflow_contract.
    def exitErp_workflow_contract(self, ctx:apgParser.Erp_workflow_contractContext):
        pass


    # Enter a parse tree produced by apgParser#erp_workflow_member.
    def enterErp_workflow_member(self, ctx:apgParser.Erp_workflow_memberContext):
        pass

    # Exit a parse tree produced by apgParser#erp_workflow_member.
    def exitErp_workflow_member(self, ctx:apgParser.Erp_workflow_memberContext):
        pass


    # Enter a parse tree produced by apgParser#erp_rule_set.
    def enterErp_rule_set(self, ctx:apgParser.Erp_rule_setContext):
        pass

    # Exit a parse tree produced by apgParser#erp_rule_set.
    def exitErp_rule_set(self, ctx:apgParser.Erp_rule_setContext):
        pass


    # Enter a parse tree produced by apgParser#erp_rule_group.
    def enterErp_rule_group(self, ctx:apgParser.Erp_rule_groupContext):
        pass

    # Exit a parse tree produced by apgParser#erp_rule_group.
    def exitErp_rule_group(self, ctx:apgParser.Erp_rule_groupContext):
        pass


    # Enter a parse tree produced by apgParser#approval_contract.
    def enterApproval_contract(self, ctx:apgParser.Approval_contractContext):
        pass

    # Exit a parse tree produced by apgParser#approval_contract.
    def exitApproval_contract(self, ctx:apgParser.Approval_contractContext):
        pass


    # Enter a parse tree produced by apgParser#approval_member.
    def enterApproval_member(self, ctx:apgParser.Approval_memberContext):
        pass

    # Exit a parse tree produced by apgParser#approval_member.
    def exitApproval_member(self, ctx:apgParser.Approval_memberContext):
        pass


    # Enter a parse tree produced by apgParser#permission_contract.
    def enterPermission_contract(self, ctx:apgParser.Permission_contractContext):
        pass

    # Exit a parse tree produced by apgParser#permission_contract.
    def exitPermission_contract(self, ctx:apgParser.Permission_contractContext):
        pass


    # Enter a parse tree produced by apgParser#permission_member.
    def enterPermission_member(self, ctx:apgParser.Permission_memberContext):
        pass

    # Exit a parse tree produced by apgParser#permission_member.
    def exitPermission_member(self, ctx:apgParser.Permission_memberContext):
        pass


    # Enter a parse tree produced by apgParser#audit_contract.
    def enterAudit_contract(self, ctx:apgParser.Audit_contractContext):
        pass

    # Exit a parse tree produced by apgParser#audit_contract.
    def exitAudit_contract(self, ctx:apgParser.Audit_contractContext):
        pass


    # Enter a parse tree produced by apgParser#audit_member.
    def enterAudit_member(self, ctx:apgParser.Audit_memberContext):
        pass

    # Exit a parse tree produced by apgParser#audit_member.
    def exitAudit_member(self, ctx:apgParser.Audit_memberContext):
        pass


    # Enter a parse tree produced by apgParser#effective_date_contract.
    def enterEffective_date_contract(self, ctx:apgParser.Effective_date_contractContext):
        pass

    # Exit a parse tree produced by apgParser#effective_date_contract.
    def exitEffective_date_contract(self, ctx:apgParser.Effective_date_contractContext):
        pass


    # Enter a parse tree produced by apgParser#effective_date_member.
    def enterEffective_date_member(self, ctx:apgParser.Effective_date_memberContext):
        pass

    # Exit a parse tree produced by apgParser#effective_date_member.
    def exitEffective_date_member(self, ctx:apgParser.Effective_date_memberContext):
        pass


    # Enter a parse tree produced by apgParser#master_data_contract.
    def enterMaster_data_contract(self, ctx:apgParser.Master_data_contractContext):
        pass

    # Exit a parse tree produced by apgParser#master_data_contract.
    def exitMaster_data_contract(self, ctx:apgParser.Master_data_contractContext):
        pass


    # Enter a parse tree produced by apgParser#master_data_member.
    def enterMaster_data_member(self, ctx:apgParser.Master_data_memberContext):
        pass

    # Exit a parse tree produced by apgParser#master_data_member.
    def exitMaster_data_member(self, ctx:apgParser.Master_data_memberContext):
        pass


    # Enter a parse tree produced by apgParser#agent_composition_block.
    def enterAgent_composition_block(self, ctx:apgParser.Agent_composition_blockContext):
        pass

    # Exit a parse tree produced by apgParser#agent_composition_block.
    def exitAgent_composition_block(self, ctx:apgParser.Agent_composition_blockContext):
        pass


    # Enter a parse tree produced by apgParser#agent_set.
    def enterAgent_set(self, ctx:apgParser.Agent_setContext):
        pass

    # Exit a parse tree produced by apgParser#agent_set.
    def exitAgent_set(self, ctx:apgParser.Agent_setContext):
        pass


    # Enter a parse tree produced by apgParser#agent_binding.
    def enterAgent_binding(self, ctx:apgParser.Agent_bindingContext):
        pass

    # Exit a parse tree produced by apgParser#agent_binding.
    def exitAgent_binding(self, ctx:apgParser.Agent_bindingContext):
        pass


    # Enter a parse tree produced by apgParser#agent_contract_member.
    def enterAgent_contract_member(self, ctx:apgParser.Agent_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#agent_contract_member.
    def exitAgent_contract_member(self, ctx:apgParser.Agent_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#agent_runtime_set.
    def enterAgent_runtime_set(self, ctx:apgParser.Agent_runtime_setContext):
        pass

    # Exit a parse tree produced by apgParser#agent_runtime_set.
    def exitAgent_runtime_set(self, ctx:apgParser.Agent_runtime_setContext):
        pass


    # Enter a parse tree produced by apgParser#agent_runtime_contract_member.
    def enterAgent_runtime_contract_member(self, ctx:apgParser.Agent_runtime_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#agent_runtime_contract_member.
    def exitAgent_runtime_contract_member(self, ctx:apgParser.Agent_runtime_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#agent_runtime_ref.
    def enterAgent_runtime_ref(self, ctx:apgParser.Agent_runtime_refContext):
        pass

    # Exit a parse tree produced by apgParser#agent_runtime_ref.
    def exitAgent_runtime_ref(self, ctx:apgParser.Agent_runtime_refContext):
        pass


    # Enter a parse tree produced by apgParser#agent_tool_set.
    def enterAgent_tool_set(self, ctx:apgParser.Agent_tool_setContext):
        pass

    # Exit a parse tree produced by apgParser#agent_tool_set.
    def exitAgent_tool_set(self, ctx:apgParser.Agent_tool_setContext):
        pass


    # Enter a parse tree produced by apgParser#agent_tool_contract_member.
    def enterAgent_tool_contract_member(self, ctx:apgParser.Agent_tool_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#agent_tool_contract_member.
    def exitAgent_tool_contract_member(self, ctx:apgParser.Agent_tool_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#agent_memory_contract.
    def enterAgent_memory_contract(self, ctx:apgParser.Agent_memory_contractContext):
        pass

    # Exit a parse tree produced by apgParser#agent_memory_contract.
    def exitAgent_memory_contract(self, ctx:apgParser.Agent_memory_contractContext):
        pass


    # Enter a parse tree produced by apgParser#handoff_graph.
    def enterHandoff_graph(self, ctx:apgParser.Handoff_graphContext):
        pass

    # Exit a parse tree produced by apgParser#handoff_graph.
    def exitHandoff_graph(self, ctx:apgParser.Handoff_graphContext):
        pass


    # Enter a parse tree produced by apgParser#handoff_edge.
    def enterHandoff_edge(self, ctx:apgParser.Handoff_edgeContext):
        pass

    # Exit a parse tree produced by apgParser#handoff_edge.
    def exitHandoff_edge(self, ctx:apgParser.Handoff_edgeContext):
        pass


    # Enter a parse tree produced by apgParser#handoff_modifier.
    def enterHandoff_modifier(self, ctx:apgParser.Handoff_modifierContext):
        pass

    # Exit a parse tree produced by apgParser#handoff_modifier.
    def exitHandoff_modifier(self, ctx:apgParser.Handoff_modifierContext):
        pass


    # Enter a parse tree produced by apgParser#model_chain.
    def enterModel_chain(self, ctx:apgParser.Model_chainContext):
        pass

    # Exit a parse tree produced by apgParser#model_chain.
    def exitModel_chain(self, ctx:apgParser.Model_chainContext):
        pass


    # Enter a parse tree produced by apgParser#model_ref.
    def enterModel_ref(self, ctx:apgParser.Model_refContext):
        pass

    # Exit a parse tree produced by apgParser#model_ref.
    def exitModel_ref(self, ctx:apgParser.Model_refContext):
        pass


    # Enter a parse tree produced by apgParser#capability_ref.
    def enterCapability_ref(self, ctx:apgParser.Capability_refContext):
        pass

    # Exit a parse tree produced by apgParser#capability_ref.
    def exitCapability_ref(self, ctx:apgParser.Capability_refContext):
        pass


    # Enter a parse tree produced by apgParser#io_contract.
    def enterIo_contract(self, ctx:apgParser.Io_contractContext):
        pass

    # Exit a parse tree produced by apgParser#io_contract.
    def exitIo_contract(self, ctx:apgParser.Io_contractContext):
        pass


    # Enter a parse tree produced by apgParser#rule_engine_block.
    def enterRule_engine_block(self, ctx:apgParser.Rule_engine_blockContext):
        pass

    # Exit a parse tree produced by apgParser#rule_engine_block.
    def exitRule_engine_block(self, ctx:apgParser.Rule_engine_blockContext):
        pass


    # Enter a parse tree produced by apgParser#rule_engine_contract.
    def enterRule_engine_contract(self, ctx:apgParser.Rule_engine_contractContext):
        pass

    # Exit a parse tree produced by apgParser#rule_engine_contract.
    def exitRule_engine_contract(self, ctx:apgParser.Rule_engine_contractContext):
        pass


    # Enter a parse tree produced by apgParser#rule_engine_member.
    def enterRule_engine_member(self, ctx:apgParser.Rule_engine_memberContext):
        pass

    # Exit a parse tree produced by apgParser#rule_engine_member.
    def exitRule_engine_member(self, ctx:apgParser.Rule_engine_memberContext):
        pass


    # Enter a parse tree produced by apgParser#rule_engine_type.
    def enterRule_engine_type(self, ctx:apgParser.Rule_engine_typeContext):
        pass

    # Exit a parse tree produced by apgParser#rule_engine_type.
    def exitRule_engine_type(self, ctx:apgParser.Rule_engine_typeContext):
        pass


    # Enter a parse tree produced by apgParser#rule_list.
    def enterRule_list(self, ctx:apgParser.Rule_listContext):
        pass

    # Exit a parse tree produced by apgParser#rule_list.
    def exitRule_list(self, ctx:apgParser.Rule_listContext):
        pass


    # Enter a parse tree produced by apgParser#rule_contract.
    def enterRule_contract(self, ctx:apgParser.Rule_contractContext):
        pass

    # Exit a parse tree produced by apgParser#rule_contract.
    def exitRule_contract(self, ctx:apgParser.Rule_contractContext):
        pass


    # Enter a parse tree produced by apgParser#rule_contract_member.
    def enterRule_contract_member(self, ctx:apgParser.Rule_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#rule_contract_member.
    def exitRule_contract_member(self, ctx:apgParser.Rule_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#rule_decision.
    def enterRule_decision(self, ctx:apgParser.Rule_decisionContext):
        pass

    # Exit a parse tree produced by apgParser#rule_decision.
    def exitRule_decision(self, ctx:apgParser.Rule_decisionContext):
        pass


    # Enter a parse tree produced by apgParser#ui_contract_block.
    def enterUi_contract_block(self, ctx:apgParser.Ui_contract_blockContext):
        pass

    # Exit a parse tree produced by apgParser#ui_contract_block.
    def exitUi_contract_block(self, ctx:apgParser.Ui_contract_blockContext):
        pass


    # Enter a parse tree produced by apgParser#ui_contract.
    def enterUi_contract(self, ctx:apgParser.Ui_contractContext):
        pass

    # Exit a parse tree produced by apgParser#ui_contract.
    def exitUi_contract(self, ctx:apgParser.Ui_contractContext):
        pass


    # Enter a parse tree produced by apgParser#ui_contract_member.
    def enterUi_contract_member(self, ctx:apgParser.Ui_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#ui_contract_member.
    def exitUi_contract_member(self, ctx:apgParser.Ui_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#screen_contract_block.
    def enterScreen_contract_block(self, ctx:apgParser.Screen_contract_blockContext):
        pass

    # Exit a parse tree produced by apgParser#screen_contract_block.
    def exitScreen_contract_block(self, ctx:apgParser.Screen_contract_blockContext):
        pass


    # Enter a parse tree produced by apgParser#screen_set.
    def enterScreen_set(self, ctx:apgParser.Screen_setContext):
        pass

    # Exit a parse tree produced by apgParser#screen_set.
    def exitScreen_set(self, ctx:apgParser.Screen_setContext):
        pass


    # Enter a parse tree produced by apgParser#screen_binding.
    def enterScreen_binding(self, ctx:apgParser.Screen_bindingContext):
        pass

    # Exit a parse tree produced by apgParser#screen_binding.
    def exitScreen_binding(self, ctx:apgParser.Screen_bindingContext):
        pass


    # Enter a parse tree produced by apgParser#screen_key.
    def enterScreen_key(self, ctx:apgParser.Screen_keyContext):
        pass

    # Exit a parse tree produced by apgParser#screen_key.
    def exitScreen_key(self, ctx:apgParser.Screen_keyContext):
        pass


    # Enter a parse tree produced by apgParser#screen_contract.
    def enterScreen_contract(self, ctx:apgParser.Screen_contractContext):
        pass

    # Exit a parse tree produced by apgParser#screen_contract.
    def exitScreen_contract(self, ctx:apgParser.Screen_contractContext):
        pass


    # Enter a parse tree produced by apgParser#screen_contract_member.
    def enterScreen_contract_member(self, ctx:apgParser.Screen_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#screen_contract_member.
    def exitScreen_contract_member(self, ctx:apgParser.Screen_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#screen_layout.
    def enterScreen_layout(self, ctx:apgParser.Screen_layoutContext):
        pass

    # Exit a parse tree produced by apgParser#screen_layout.
    def exitScreen_layout(self, ctx:apgParser.Screen_layoutContext):
        pass


    # Enter a parse tree produced by apgParser#screen_element_list.
    def enterScreen_element_list(self, ctx:apgParser.Screen_element_listContext):
        pass

    # Exit a parse tree produced by apgParser#screen_element_list.
    def exitScreen_element_list(self, ctx:apgParser.Screen_element_listContext):
        pass


    # Enter a parse tree produced by apgParser#screen_element_ref.
    def enterScreen_element_ref(self, ctx:apgParser.Screen_element_refContext):
        pass

    # Exit a parse tree produced by apgParser#screen_element_ref.
    def exitScreen_element_ref(self, ctx:apgParser.Screen_element_refContext):
        pass


    # Enter a parse tree produced by apgParser#screen_element_member.
    def enterScreen_element_member(self, ctx:apgParser.Screen_element_memberContext):
        pass

    # Exit a parse tree produced by apgParser#screen_element_member.
    def exitScreen_element_member(self, ctx:apgParser.Screen_element_memberContext):
        pass


    # Enter a parse tree produced by apgParser#screen_event_list.
    def enterScreen_event_list(self, ctx:apgParser.Screen_event_listContext):
        pass

    # Exit a parse tree produced by apgParser#screen_event_list.
    def exitScreen_event_list(self, ctx:apgParser.Screen_event_listContext):
        pass


    # Enter a parse tree produced by apgParser#screen_event.
    def enterScreen_event(self, ctx:apgParser.Screen_eventContext):
        pass

    # Exit a parse tree produced by apgParser#screen_event.
    def exitScreen_event(self, ctx:apgParser.Screen_eventContext):
        pass


    # Enter a parse tree produced by apgParser#screen_event_member.
    def enterScreen_event_member(self, ctx:apgParser.Screen_event_memberContext):
        pass

    # Exit a parse tree produced by apgParser#screen_event_member.
    def exitScreen_event_member(self, ctx:apgParser.Screen_event_memberContext):
        pass


    # Enter a parse tree produced by apgParser#screen_relationship_list.
    def enterScreen_relationship_list(self, ctx:apgParser.Screen_relationship_listContext):
        pass

    # Exit a parse tree produced by apgParser#screen_relationship_list.
    def exitScreen_relationship_list(self, ctx:apgParser.Screen_relationship_listContext):
        pass


    # Enter a parse tree produced by apgParser#screen_relationship.
    def enterScreen_relationship(self, ctx:apgParser.Screen_relationshipContext):
        pass

    # Exit a parse tree produced by apgParser#screen_relationship.
    def exitScreen_relationship(self, ctx:apgParser.Screen_relationshipContext):
        pass


    # Enter a parse tree produced by apgParser#screen_relation_edge.
    def enterScreen_relation_edge(self, ctx:apgParser.Screen_relation_edgeContext):
        pass

    # Exit a parse tree produced by apgParser#screen_relation_edge.
    def exitScreen_relation_edge(self, ctx:apgParser.Screen_relation_edgeContext):
        pass


    # Enter a parse tree produced by apgParser#screen_relation_modifier.
    def enterScreen_relation_modifier(self, ctx:apgParser.Screen_relation_modifierContext):
        pass

    # Exit a parse tree produced by apgParser#screen_relation_modifier.
    def exitScreen_relation_modifier(self, ctx:apgParser.Screen_relation_modifierContext):
        pass


    # Enter a parse tree produced by apgParser#screen_relationship_member.
    def enterScreen_relationship_member(self, ctx:apgParser.Screen_relationship_memberContext):
        pass

    # Exit a parse tree produced by apgParser#screen_relationship_member.
    def exitScreen_relationship_member(self, ctx:apgParser.Screen_relationship_memberContext):
        pass


    # Enter a parse tree produced by apgParser#ui_shell.
    def enterUi_shell(self, ctx:apgParser.Ui_shellContext):
        pass

    # Exit a parse tree produced by apgParser#ui_shell.
    def exitUi_shell(self, ctx:apgParser.Ui_shellContext):
        pass


    # Enter a parse tree produced by apgParser#ui_route_list.
    def enterUi_route_list(self, ctx:apgParser.Ui_route_listContext):
        pass

    # Exit a parse tree produced by apgParser#ui_route_list.
    def exitUi_route_list(self, ctx:apgParser.Ui_route_listContext):
        pass


    # Enter a parse tree produced by apgParser#ui_route.
    def enterUi_route(self, ctx:apgParser.Ui_routeContext):
        pass

    # Exit a parse tree produced by apgParser#ui_route.
    def exitUi_route(self, ctx:apgParser.Ui_routeContext):
        pass


    # Enter a parse tree produced by apgParser#ui_route_member.
    def enterUi_route_member(self, ctx:apgParser.Ui_route_memberContext):
        pass

    # Exit a parse tree produced by apgParser#ui_route_member.
    def exitUi_route_member(self, ctx:apgParser.Ui_route_memberContext):
        pass


    # Enter a parse tree produced by apgParser#theme_contract_block.
    def enterTheme_contract_block(self, ctx:apgParser.Theme_contract_blockContext):
        pass

    # Exit a parse tree produced by apgParser#theme_contract_block.
    def exitTheme_contract_block(self, ctx:apgParser.Theme_contract_blockContext):
        pass


    # Enter a parse tree produced by apgParser#theme_contract.
    def enterTheme_contract(self, ctx:apgParser.Theme_contractContext):
        pass

    # Exit a parse tree produced by apgParser#theme_contract.
    def exitTheme_contract(self, ctx:apgParser.Theme_contractContext):
        pass


    # Enter a parse tree produced by apgParser#theme_contract_member.
    def enterTheme_contract_member(self, ctx:apgParser.Theme_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#theme_contract_member.
    def exitTheme_contract_member(self, ctx:apgParser.Theme_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#theme_token_map.
    def enterTheme_token_map(self, ctx:apgParser.Theme_token_mapContext):
        pass

    # Exit a parse tree produced by apgParser#theme_token_map.
    def exitTheme_token_map(self, ctx:apgParser.Theme_token_mapContext):
        pass


    # Enter a parse tree produced by apgParser#theme_token.
    def enterTheme_token(self, ctx:apgParser.Theme_tokenContext):
        pass

    # Exit a parse tree produced by apgParser#theme_token.
    def exitTheme_token(self, ctx:apgParser.Theme_tokenContext):
        pass


    # Enter a parse tree produced by apgParser#runtime_contract.
    def enterRuntime_contract(self, ctx:apgParser.Runtime_contractContext):
        pass

    # Exit a parse tree produced by apgParser#runtime_contract.
    def exitRuntime_contract(self, ctx:apgParser.Runtime_contractContext):
        pass


    # Enter a parse tree produced by apgParser#runtime_contract_member.
    def enterRuntime_contract_member(self, ctx:apgParser.Runtime_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#runtime_contract_member.
    def exitRuntime_contract_member(self, ctx:apgParser.Runtime_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#runtime_backend.
    def enterRuntime_backend(self, ctx:apgParser.Runtime_backendContext):
        pass

    # Exit a parse tree produced by apgParser#runtime_backend.
    def exitRuntime_backend(self, ctx:apgParser.Runtime_backendContext):
        pass


    # Enter a parse tree produced by apgParser#stream_runtime_block.
    def enterStream_runtime_block(self, ctx:apgParser.Stream_runtime_blockContext):
        pass

    # Exit a parse tree produced by apgParser#stream_runtime_block.
    def exitStream_runtime_block(self, ctx:apgParser.Stream_runtime_blockContext):
        pass


    # Enter a parse tree produced by apgParser#stream_runtime_contract.
    def enterStream_runtime_contract(self, ctx:apgParser.Stream_runtime_contractContext):
        pass

    # Exit a parse tree produced by apgParser#stream_runtime_contract.
    def exitStream_runtime_contract(self, ctx:apgParser.Stream_runtime_contractContext):
        pass


    # Enter a parse tree produced by apgParser#stream_runtime_member.
    def enterStream_runtime_member(self, ctx:apgParser.Stream_runtime_memberContext):
        pass

    # Exit a parse tree produced by apgParser#stream_runtime_member.
    def exitStream_runtime_member(self, ctx:apgParser.Stream_runtime_memberContext):
        pass


    # Enter a parse tree produced by apgParser#stream_processor.
    def enterStream_processor(self, ctx:apgParser.Stream_processorContext):
        pass

    # Exit a parse tree produced by apgParser#stream_processor.
    def exitStream_processor(self, ctx:apgParser.Stream_processorContext):
        pass


    # Enter a parse tree produced by apgParser#i18n_contract_block.
    def enterI18n_contract_block(self, ctx:apgParser.I18n_contract_blockContext):
        pass

    # Exit a parse tree produced by apgParser#i18n_contract_block.
    def exitI18n_contract_block(self, ctx:apgParser.I18n_contract_blockContext):
        pass


    # Enter a parse tree produced by apgParser#i18n_contract.
    def enterI18n_contract(self, ctx:apgParser.I18n_contractContext):
        pass

    # Exit a parse tree produced by apgParser#i18n_contract.
    def exitI18n_contract(self, ctx:apgParser.I18n_contractContext):
        pass


    # Enter a parse tree produced by apgParser#i18n_contract_member.
    def enterI18n_contract_member(self, ctx:apgParser.I18n_contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#i18n_contract_member.
    def exitI18n_contract_member(self, ctx:apgParser.I18n_contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#language_collection.
    def enterLanguage_collection(self, ctx:apgParser.Language_collectionContext):
        pass

    # Exit a parse tree produced by apgParser#language_collection.
    def exitLanguage_collection(self, ctx:apgParser.Language_collectionContext):
        pass


    # Enter a parse tree produced by apgParser#language_list.
    def enterLanguage_list(self, ctx:apgParser.Language_listContext):
        pass

    # Exit a parse tree produced by apgParser#language_list.
    def exitLanguage_list(self, ctx:apgParser.Language_listContext):
        pass


    # Enter a parse tree produced by apgParser#language_code.
    def enterLanguage_code(self, ctx:apgParser.Language_codeContext):
        pass

    # Exit a parse tree produced by apgParser#language_code.
    def exitLanguage_code(self, ctx:apgParser.Language_codeContext):
        pass


    # Enter a parse tree produced by apgParser#reference_list.
    def enterReference_list(self, ctx:apgParser.Reference_listContext):
        pass

    # Exit a parse tree produced by apgParser#reference_list.
    def exitReference_list(self, ctx:apgParser.Reference_listContext):
        pass


    # Enter a parse tree produced by apgParser#contract_object.
    def enterContract_object(self, ctx:apgParser.Contract_objectContext):
        pass

    # Exit a parse tree produced by apgParser#contract_object.
    def exitContract_object(self, ctx:apgParser.Contract_objectContext):
        pass


    # Enter a parse tree produced by apgParser#contract_member.
    def enterContract_member(self, ctx:apgParser.Contract_memberContext):
        pass

    # Exit a parse tree produced by apgParser#contract_member.
    def exitContract_member(self, ctx:apgParser.Contract_memberContext):
        pass


    # Enter a parse tree produced by apgParser#contract_value.
    def enterContract_value(self, ctx:apgParser.Contract_valueContext):
        pass

    # Exit a parse tree produced by apgParser#contract_value.
    def exitContract_value(self, ctx:apgParser.Contract_valueContext):
        pass


    # Enter a parse tree produced by apgParser#contract_array.
    def enterContract_array(self, ctx:apgParser.Contract_arrayContext):
        pass

    # Exit a parse tree produced by apgParser#contract_array.
    def exitContract_array(self, ctx:apgParser.Contract_arrayContext):
        pass


    # Enter a parse tree produced by apgParser#contract_scalar.
    def enterContract_scalar(self, ctx:apgParser.Contract_scalarContext):
        pass

    # Exit a parse tree produced by apgParser#contract_scalar.
    def exitContract_scalar(self, ctx:apgParser.Contract_scalarContext):
        pass


    # Enter a parse tree produced by apgParser#contract_separator.
    def enterContract_separator(self, ctx:apgParser.Contract_separatorContext):
        pass

    # Exit a parse tree produced by apgParser#contract_separator.
    def exitContract_separator(self, ctx:apgParser.Contract_separatorContext):
        pass


    # Enter a parse tree produced by apgParser#type_annotation.
    def enterType_annotation(self, ctx:apgParser.Type_annotationContext):
        pass

    # Exit a parse tree produced by apgParser#type_annotation.
    def exitType_annotation(self, ctx:apgParser.Type_annotationContext):
        pass


    # Enter a parse tree produced by apgParser#union_type.
    def enterUnion_type(self, ctx:apgParser.Union_typeContext):
        pass

    # Exit a parse tree produced by apgParser#union_type.
    def exitUnion_type(self, ctx:apgParser.Union_typeContext):
        pass


    # Enter a parse tree produced by apgParser#primary_type.
    def enterPrimary_type(self, ctx:apgParser.Primary_typeContext):
        pass

    # Exit a parse tree produced by apgParser#primary_type.
    def exitPrimary_type(self, ctx:apgParser.Primary_typeContext):
        pass


    # Enter a parse tree produced by apgParser#basic_type.
    def enterBasic_type(self, ctx:apgParser.Basic_typeContext):
        pass

    # Exit a parse tree produced by apgParser#basic_type.
    def exitBasic_type(self, ctx:apgParser.Basic_typeContext):
        pass


    # Enter a parse tree produced by apgParser#optional_suffix.
    def enterOptional_suffix(self, ctx:apgParser.Optional_suffixContext):
        pass

    # Exit a parse tree produced by apgParser#optional_suffix.
    def exitOptional_suffix(self, ctx:apgParser.Optional_suffixContext):
        pass


    # Enter a parse tree produced by apgParser#generic_type.
    def enterGeneric_type(self, ctx:apgParser.Generic_typeContext):
        pass

    # Exit a parse tree produced by apgParser#generic_type.
    def exitGeneric_type(self, ctx:apgParser.Generic_typeContext):
        pass


    # Enter a parse tree produced by apgParser#list_type.
    def enterList_type(self, ctx:apgParser.List_typeContext):
        pass

    # Exit a parse tree produced by apgParser#list_type.
    def exitList_type(self, ctx:apgParser.List_typeContext):
        pass


    # Enter a parse tree produced by apgParser#dict_type.
    def enterDict_type(self, ctx:apgParser.Dict_typeContext):
        pass

    # Exit a parse tree produced by apgParser#dict_type.
    def exitDict_type(self, ctx:apgParser.Dict_typeContext):
        pass


    # Enter a parse tree produced by apgParser#value_expr.
    def enterValue_expr(self, ctx:apgParser.Value_exprContext):
        pass

    # Exit a parse tree produced by apgParser#value_expr.
    def exitValue_expr(self, ctx:apgParser.Value_exprContext):
        pass


    # Enter a parse tree produced by apgParser#simple_value.
    def enterSimple_value(self, ctx:apgParser.Simple_valueContext):
        pass

    # Exit a parse tree produced by apgParser#simple_value.
    def exitSimple_value(self, ctx:apgParser.Simple_valueContext):
        pass


    # Enter a parse tree produced by apgParser#env_var.
    def enterEnv_var(self, ctx:apgParser.Env_varContext):
        pass

    # Exit a parse tree produced by apgParser#env_var.
    def exitEnv_var(self, ctx:apgParser.Env_varContext):
        pass


    # Enter a parse tree produced by apgParser#f_string.
    def enterF_string(self, ctx:apgParser.F_stringContext):
        pass

    # Exit a parse tree produced by apgParser#f_string.
    def exitF_string(self, ctx:apgParser.F_stringContext):
        pass


    # Enter a parse tree produced by apgParser#list_value.
    def enterList_value(self, ctx:apgParser.List_valueContext):
        pass

    # Exit a parse tree produced by apgParser#list_value.
    def exitList_value(self, ctx:apgParser.List_valueContext):
        pass


    # Enter a parse tree produced by apgParser#dict_value.
    def enterDict_value(self, ctx:apgParser.Dict_valueContext):
        pass

    # Exit a parse tree produced by apgParser#dict_value.
    def exitDict_value(self, ctx:apgParser.Dict_valueContext):
        pass


    # Enter a parse tree produced by apgParser#key_value_pair.
    def enterKey_value_pair(self, ctx:apgParser.Key_value_pairContext):
        pass

    # Exit a parse tree produced by apgParser#key_value_pair.
    def exitKey_value_pair(self, ctx:apgParser.Key_value_pairContext):
        pass


    # Enter a parse tree produced by apgParser#cascade_value.
    def enterCascade_value(self, ctx:apgParser.Cascade_valueContext):
        pass

    # Exit a parse tree produced by apgParser#cascade_value.
    def exitCascade_value(self, ctx:apgParser.Cascade_valueContext):
        pass


    # Enter a parse tree produced by apgParser#fallback_chain.
    def enterFallback_chain(self, ctx:apgParser.Fallback_chainContext):
        pass

    # Exit a parse tree produced by apgParser#fallback_chain.
    def exitFallback_chain(self, ctx:apgParser.Fallback_chainContext):
        pass


    # Enter a parse tree produced by apgParser#physical_literal.
    def enterPhysical_literal(self, ctx:apgParser.Physical_literalContext):
        pass

    # Exit a parse tree produced by apgParser#physical_literal.
    def exitPhysical_literal(self, ctx:apgParser.Physical_literalContext):
        pass


    # Enter a parse tree produced by apgParser#agent_memory_value.
    def enterAgent_memory_value(self, ctx:apgParser.Agent_memory_valueContext):
        pass

    # Exit a parse tree produced by apgParser#agent_memory_value.
    def exitAgent_memory_value(self, ctx:apgParser.Agent_memory_valueContext):
        pass


    # Enter a parse tree produced by apgParser#reference_value.
    def enterReference_value(self, ctx:apgParser.Reference_valueContext):
        pass

    # Exit a parse tree produced by apgParser#reference_value.
    def exitReference_value(self, ctx:apgParser.Reference_valueContext):
        pass


    # Enter a parse tree produced by apgParser#combination_expr.
    def enterCombination_expr(self, ctx:apgParser.Combination_exprContext):
        pass

    # Exit a parse tree produced by apgParser#combination_expr.
    def exitCombination_expr(self, ctx:apgParser.Combination_exprContext):
        pass


    # Enter a parse tree produced by apgParser#url_pattern.
    def enterUrl_pattern(self, ctx:apgParser.Url_patternContext):
        pass

    # Exit a parse tree produced by apgParser#url_pattern.
    def exitUrl_pattern(self, ctx:apgParser.Url_patternContext):
        pass


    # Enter a parse tree produced by apgParser#regex_pattern.
    def enterRegex_pattern(self, ctx:apgParser.Regex_patternContext):
        pass

    # Exit a parse tree produced by apgParser#regex_pattern.
    def exitRegex_pattern(self, ctx:apgParser.Regex_patternContext):
        pass


    # Enter a parse tree produced by apgParser#time_expr.
    def enterTime_expr(self, ctx:apgParser.Time_exprContext):
        pass

    # Exit a parse tree produced by apgParser#time_expr.
    def exitTime_expr(self, ctx:apgParser.Time_exprContext):
        pass


    # Enter a parse tree produced by apgParser#async_expr.
    def enterAsync_expr(self, ctx:apgParser.Async_exprContext):
        pass

    # Exit a parse tree produced by apgParser#async_expr.
    def exitAsync_expr(self, ctx:apgParser.Async_exprContext):
        pass


    # Enter a parse tree produced by apgParser#behavior_item.
    def enterBehavior_item(self, ctx:apgParser.Behavior_itemContext):
        pass

    # Exit a parse tree produced by apgParser#behavior_item.
    def exitBehavior_item(self, ctx:apgParser.Behavior_itemContext):
        pass


    # Enter a parse tree produced by apgParser#annotation.
    def enterAnnotation(self, ctx:apgParser.AnnotationContext):
        pass

    # Exit a parse tree produced by apgParser#annotation.
    def exitAnnotation(self, ctx:apgParser.AnnotationContext):
        pass


    # Enter a parse tree produced by apgParser#annotation_body.
    def enterAnnotation_body(self, ctx:apgParser.Annotation_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#annotation_body.
    def exitAnnotation_body(self, ctx:apgParser.Annotation_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#annotation_member.
    def enterAnnotation_member(self, ctx:apgParser.Annotation_memberContext):
        pass

    # Exit a parse tree produced by apgParser#annotation_member.
    def exitAnnotation_member(self, ctx:apgParser.Annotation_memberContext):
        pass


    # Enter a parse tree produced by apgParser#nested_annotation.
    def enterNested_annotation(self, ctx:apgParser.Nested_annotationContext):
        pass

    # Exit a parse tree produced by apgParser#nested_annotation.
    def exitNested_annotation(self, ctx:apgParser.Nested_annotationContext):
        pass


    # Enter a parse tree produced by apgParser#when_clause.
    def enterWhen_clause(self, ctx:apgParser.When_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#when_clause.
    def exitWhen_clause(self, ctx:apgParser.When_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#then_clause.
    def enterThen_clause(self, ctx:apgParser.Then_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#then_clause.
    def exitThen_clause(self, ctx:apgParser.Then_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#method_def.
    def enterMethod_def(self, ctx:apgParser.Method_defContext):
        pass

    # Exit a parse tree produced by apgParser#method_def.
    def exitMethod_def(self, ctx:apgParser.Method_defContext):
        pass


    # Enter a parse tree produced by apgParser#async_modifier.
    def enterAsync_modifier(self, ctx:apgParser.Async_modifierContext):
        pass

    # Exit a parse tree produced by apgParser#async_modifier.
    def exitAsync_modifier(self, ctx:apgParser.Async_modifierContext):
        pass


    # Enter a parse tree produced by apgParser#param_list.
    def enterParam_list(self, ctx:apgParser.Param_listContext):
        pass

    # Exit a parse tree produced by apgParser#param_list.
    def exitParam_list(self, ctx:apgParser.Param_listContext):
        pass


    # Enter a parse tree produced by apgParser#parameter.
    def enterParameter(self, ctx:apgParser.ParameterContext):
        pass

    # Exit a parse tree produced by apgParser#parameter.
    def exitParameter(self, ctx:apgParser.ParameterContext):
        pass


    # Enter a parse tree produced by apgParser#return_type.
    def enterReturn_type(self, ctx:apgParser.Return_typeContext):
        pass

    # Exit a parse tree produced by apgParser#return_type.
    def exitReturn_type(self, ctx:apgParser.Return_typeContext):
        pass


    # Enter a parse tree produced by apgParser#method_body.
    def enterMethod_body(self, ctx:apgParser.Method_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#method_body.
    def exitMethod_body(self, ctx:apgParser.Method_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#statement.
    def enterStatement(self, ctx:apgParser.StatementContext):
        pass

    # Exit a parse tree produced by apgParser#statement.
    def exitStatement(self, ctx:apgParser.StatementContext):
        pass


    # Enter a parse tree produced by apgParser#simple_statement.
    def enterSimple_statement(self, ctx:apgParser.Simple_statementContext):
        pass

    # Exit a parse tree produced by apgParser#simple_statement.
    def exitSimple_statement(self, ctx:apgParser.Simple_statementContext):
        pass


    # Enter a parse tree produced by apgParser#compound_statement.
    def enterCompound_statement(self, ctx:apgParser.Compound_statementContext):
        pass

    # Exit a parse tree produced by apgParser#compound_statement.
    def exitCompound_statement(self, ctx:apgParser.Compound_statementContext):
        pass


    # Enter a parse tree produced by apgParser#assignment.
    def enterAssignment(self, ctx:apgParser.AssignmentContext):
        pass

    # Exit a parse tree produced by apgParser#assignment.
    def exitAssignment(self, ctx:apgParser.AssignmentContext):
        pass


    # Enter a parse tree produced by apgParser#method_call.
    def enterMethod_call(self, ctx:apgParser.Method_callContext):
        pass

    # Exit a parse tree produced by apgParser#method_call.
    def exitMethod_call(self, ctx:apgParser.Method_callContext):
        pass


    # Enter a parse tree produced by apgParser#args.
    def enterArgs(self, ctx:apgParser.ArgsContext):
        pass

    # Exit a parse tree produced by apgParser#args.
    def exitArgs(self, ctx:apgParser.ArgsContext):
        pass


    # Enter a parse tree produced by apgParser#argument.
    def enterArgument(self, ctx:apgParser.ArgumentContext):
        pass

    # Exit a parse tree produced by apgParser#argument.
    def exitArgument(self, ctx:apgParser.ArgumentContext):
        pass


    # Enter a parse tree produced by apgParser#return_statement.
    def enterReturn_statement(self, ctx:apgParser.Return_statementContext):
        pass

    # Exit a parse tree produced by apgParser#return_statement.
    def exitReturn_statement(self, ctx:apgParser.Return_statementContext):
        pass


    # Enter a parse tree produced by apgParser#break_statement.
    def enterBreak_statement(self, ctx:apgParser.Break_statementContext):
        pass

    # Exit a parse tree produced by apgParser#break_statement.
    def exitBreak_statement(self, ctx:apgParser.Break_statementContext):
        pass


    # Enter a parse tree produced by apgParser#continue_statement.
    def enterContinue_statement(self, ctx:apgParser.Continue_statementContext):
        pass

    # Exit a parse tree produced by apgParser#continue_statement.
    def exitContinue_statement(self, ctx:apgParser.Continue_statementContext):
        pass


    # Enter a parse tree produced by apgParser#pass_statement.
    def enterPass_statement(self, ctx:apgParser.Pass_statementContext):
        pass

    # Exit a parse tree produced by apgParser#pass_statement.
    def exitPass_statement(self, ctx:apgParser.Pass_statementContext):
        pass


    # Enter a parse tree produced by apgParser#assert_statement.
    def enterAssert_statement(self, ctx:apgParser.Assert_statementContext):
        pass

    # Exit a parse tree produced by apgParser#assert_statement.
    def exitAssert_statement(self, ctx:apgParser.Assert_statementContext):
        pass


    # Enter a parse tree produced by apgParser#yield_statement.
    def enterYield_statement(self, ctx:apgParser.Yield_statementContext):
        pass

    # Exit a parse tree produced by apgParser#yield_statement.
    def exitYield_statement(self, ctx:apgParser.Yield_statementContext):
        pass


    # Enter a parse tree produced by apgParser#if_statement.
    def enterIf_statement(self, ctx:apgParser.If_statementContext):
        pass

    # Exit a parse tree produced by apgParser#if_statement.
    def exitIf_statement(self, ctx:apgParser.If_statementContext):
        pass


    # Enter a parse tree produced by apgParser#elif_clause.
    def enterElif_clause(self, ctx:apgParser.Elif_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#elif_clause.
    def exitElif_clause(self, ctx:apgParser.Elif_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#else_clause.
    def enterElse_clause(self, ctx:apgParser.Else_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#else_clause.
    def exitElse_clause(self, ctx:apgParser.Else_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#for_statement.
    def enterFor_statement(self, ctx:apgParser.For_statementContext):
        pass

    # Exit a parse tree produced by apgParser#for_statement.
    def exitFor_statement(self, ctx:apgParser.For_statementContext):
        pass


    # Enter a parse tree produced by apgParser#while_statement.
    def enterWhile_statement(self, ctx:apgParser.While_statementContext):
        pass

    # Exit a parse tree produced by apgParser#while_statement.
    def exitWhile_statement(self, ctx:apgParser.While_statementContext):
        pass


    # Enter a parse tree produced by apgParser#try_statement.
    def enterTry_statement(self, ctx:apgParser.Try_statementContext):
        pass

    # Exit a parse tree produced by apgParser#try_statement.
    def exitTry_statement(self, ctx:apgParser.Try_statementContext):
        pass


    # Enter a parse tree produced by apgParser#except_clause.
    def enterExcept_clause(self, ctx:apgParser.Except_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#except_clause.
    def exitExcept_clause(self, ctx:apgParser.Except_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#exception_spec.
    def enterException_spec(self, ctx:apgParser.Exception_specContext):
        pass

    # Exit a parse tree produced by apgParser#exception_spec.
    def exitException_spec(self, ctx:apgParser.Exception_specContext):
        pass


    # Enter a parse tree produced by apgParser#finally_clause.
    def enterFinally_clause(self, ctx:apgParser.Finally_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#finally_clause.
    def exitFinally_clause(self, ctx:apgParser.Finally_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#with_statement.
    def enterWith_statement(self, ctx:apgParser.With_statementContext):
        pass

    # Exit a parse tree produced by apgParser#with_statement.
    def exitWith_statement(self, ctx:apgParser.With_statementContext):
        pass


    # Enter a parse tree produced by apgParser#with_item.
    def enterWith_item(self, ctx:apgParser.With_itemContext):
        pass

    # Exit a parse tree produced by apgParser#with_item.
    def exitWith_item(self, ctx:apgParser.With_itemContext):
        pass


    # Enter a parse tree produced by apgParser#match_statement.
    def enterMatch_statement(self, ctx:apgParser.Match_statementContext):
        pass

    # Exit a parse tree produced by apgParser#match_statement.
    def exitMatch_statement(self, ctx:apgParser.Match_statementContext):
        pass


    # Enter a parse tree produced by apgParser#case_clause.
    def enterCase_clause(self, ctx:apgParser.Case_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#case_clause.
    def exitCase_clause(self, ctx:apgParser.Case_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#pattern.
    def enterPattern(self, ctx:apgParser.PatternContext):
        pass

    # Exit a parse tree produced by apgParser#pattern.
    def exitPattern(self, ctx:apgParser.PatternContext):
        pass


    # Enter a parse tree produced by apgParser#or_pattern.
    def enterOr_pattern(self, ctx:apgParser.Or_patternContext):
        pass

    # Exit a parse tree produced by apgParser#or_pattern.
    def exitOr_pattern(self, ctx:apgParser.Or_patternContext):
        pass


    # Enter a parse tree produced by apgParser#primary_pattern.
    def enterPrimary_pattern(self, ctx:apgParser.Primary_patternContext):
        pass

    # Exit a parse tree produced by apgParser#primary_pattern.
    def exitPrimary_pattern(self, ctx:apgParser.Primary_patternContext):
        pass


    # Enter a parse tree produced by apgParser#literal_pattern.
    def enterLiteral_pattern(self, ctx:apgParser.Literal_patternContext):
        pass

    # Exit a parse tree produced by apgParser#literal_pattern.
    def exitLiteral_pattern(self, ctx:apgParser.Literal_patternContext):
        pass


    # Enter a parse tree produced by apgParser#capture_pattern.
    def enterCapture_pattern(self, ctx:apgParser.Capture_patternContext):
        pass

    # Exit a parse tree produced by apgParser#capture_pattern.
    def exitCapture_pattern(self, ctx:apgParser.Capture_patternContext):
        pass


    # Enter a parse tree produced by apgParser#wildcard_pattern.
    def enterWildcard_pattern(self, ctx:apgParser.Wildcard_patternContext):
        pass

    # Exit a parse tree produced by apgParser#wildcard_pattern.
    def exitWildcard_pattern(self, ctx:apgParser.Wildcard_patternContext):
        pass


    # Enter a parse tree produced by apgParser#value_pattern.
    def enterValue_pattern(self, ctx:apgParser.Value_patternContext):
        pass

    # Exit a parse tree produced by apgParser#value_pattern.
    def exitValue_pattern(self, ctx:apgParser.Value_patternContext):
        pass


    # Enter a parse tree produced by apgParser#sequence_pattern.
    def enterSequence_pattern(self, ctx:apgParser.Sequence_patternContext):
        pass

    # Exit a parse tree produced by apgParser#sequence_pattern.
    def exitSequence_pattern(self, ctx:apgParser.Sequence_patternContext):
        pass


    # Enter a parse tree produced by apgParser#mapping_pattern.
    def enterMapping_pattern(self, ctx:apgParser.Mapping_patternContext):
        pass

    # Exit a parse tree produced by apgParser#mapping_pattern.
    def exitMapping_pattern(self, ctx:apgParser.Mapping_patternContext):
        pass


    # Enter a parse tree produced by apgParser#mapping_pattern_pair.
    def enterMapping_pattern_pair(self, ctx:apgParser.Mapping_pattern_pairContext):
        pass

    # Exit a parse tree produced by apgParser#mapping_pattern_pair.
    def exitMapping_pattern_pair(self, ctx:apgParser.Mapping_pattern_pairContext):
        pass


    # Enter a parse tree produced by apgParser#class_pattern.
    def enterClass_pattern(self, ctx:apgParser.Class_patternContext):
        pass

    # Exit a parse tree produced by apgParser#class_pattern.
    def exitClass_pattern(self, ctx:apgParser.Class_patternContext):
        pass


    # Enter a parse tree produced by apgParser#guard.
    def enterGuard(self, ctx:apgParser.GuardContext):
        pass

    # Exit a parse tree produced by apgParser#guard.
    def exitGuard(self, ctx:apgParser.GuardContext):
        pass


    # Enter a parse tree produced by apgParser#async_statement.
    def enterAsync_statement(self, ctx:apgParser.Async_statementContext):
        pass

    # Exit a parse tree produced by apgParser#async_statement.
    def exitAsync_statement(self, ctx:apgParser.Async_statementContext):
        pass


    # Enter a parse tree produced by apgParser#statement_block.
    def enterStatement_block(self, ctx:apgParser.Statement_blockContext):
        pass

    # Exit a parse tree produced by apgParser#statement_block.
    def exitStatement_block(self, ctx:apgParser.Statement_blockContext):
        pass


    # Enter a parse tree produced by apgParser#expression.
    def enterExpression(self, ctx:apgParser.ExpressionContext):
        pass

    # Exit a parse tree produced by apgParser#expression.
    def exitExpression(self, ctx:apgParser.ExpressionContext):
        pass


    # Enter a parse tree produced by apgParser#lambda_expr.
    def enterLambda_expr(self, ctx:apgParser.Lambda_exprContext):
        pass

    # Exit a parse tree produced by apgParser#lambda_expr.
    def exitLambda_expr(self, ctx:apgParser.Lambda_exprContext):
        pass


    # Enter a parse tree produced by apgParser#conditional_expr.
    def enterConditional_expr(self, ctx:apgParser.Conditional_exprContext):
        pass

    # Exit a parse tree produced by apgParser#conditional_expr.
    def exitConditional_expr(self, ctx:apgParser.Conditional_exprContext):
        pass


    # Enter a parse tree produced by apgParser#null_coalesce_expr.
    def enterNull_coalesce_expr(self, ctx:apgParser.Null_coalesce_exprContext):
        pass

    # Exit a parse tree produced by apgParser#null_coalesce_expr.
    def exitNull_coalesce_expr(self, ctx:apgParser.Null_coalesce_exprContext):
        pass


    # Enter a parse tree produced by apgParser#or_test.
    def enterOr_test(self, ctx:apgParser.Or_testContext):
        pass

    # Exit a parse tree produced by apgParser#or_test.
    def exitOr_test(self, ctx:apgParser.Or_testContext):
        pass


    # Enter a parse tree produced by apgParser#and_test.
    def enterAnd_test(self, ctx:apgParser.And_testContext):
        pass

    # Exit a parse tree produced by apgParser#and_test.
    def exitAnd_test(self, ctx:apgParser.And_testContext):
        pass


    # Enter a parse tree produced by apgParser#not_test.
    def enterNot_test(self, ctx:apgParser.Not_testContext):
        pass

    # Exit a parse tree produced by apgParser#not_test.
    def exitNot_test(self, ctx:apgParser.Not_testContext):
        pass


    # Enter a parse tree produced by apgParser#comparison.
    def enterComparison(self, ctx:apgParser.ComparisonContext):
        pass

    # Exit a parse tree produced by apgParser#comparison.
    def exitComparison(self, ctx:apgParser.ComparisonContext):
        pass


    # Enter a parse tree produced by apgParser#pipeline_expr.
    def enterPipeline_expr(self, ctx:apgParser.Pipeline_exprContext):
        pass

    # Exit a parse tree produced by apgParser#pipeline_expr.
    def exitPipeline_expr(self, ctx:apgParser.Pipeline_exprContext):
        pass


    # Enter a parse tree produced by apgParser#comp_op.
    def enterComp_op(self, ctx:apgParser.Comp_opContext):
        pass

    # Exit a parse tree produced by apgParser#comp_op.
    def exitComp_op(self, ctx:apgParser.Comp_opContext):
        pass


    # Enter a parse tree produced by apgParser#bitwise_or.
    def enterBitwise_or(self, ctx:apgParser.Bitwise_orContext):
        pass

    # Exit a parse tree produced by apgParser#bitwise_or.
    def exitBitwise_or(self, ctx:apgParser.Bitwise_orContext):
        pass


    # Enter a parse tree produced by apgParser#bitwise_xor.
    def enterBitwise_xor(self, ctx:apgParser.Bitwise_xorContext):
        pass

    # Exit a parse tree produced by apgParser#bitwise_xor.
    def exitBitwise_xor(self, ctx:apgParser.Bitwise_xorContext):
        pass


    # Enter a parse tree produced by apgParser#bitwise_and.
    def enterBitwise_and(self, ctx:apgParser.Bitwise_andContext):
        pass

    # Exit a parse tree produced by apgParser#bitwise_and.
    def exitBitwise_and(self, ctx:apgParser.Bitwise_andContext):
        pass


    # Enter a parse tree produced by apgParser#shift_expr.
    def enterShift_expr(self, ctx:apgParser.Shift_exprContext):
        pass

    # Exit a parse tree produced by apgParser#shift_expr.
    def exitShift_expr(self, ctx:apgParser.Shift_exprContext):
        pass


    # Enter a parse tree produced by apgParser#arith_expr.
    def enterArith_expr(self, ctx:apgParser.Arith_exprContext):
        pass

    # Exit a parse tree produced by apgParser#arith_expr.
    def exitArith_expr(self, ctx:apgParser.Arith_exprContext):
        pass


    # Enter a parse tree produced by apgParser#term.
    def enterTerm(self, ctx:apgParser.TermContext):
        pass

    # Exit a parse tree produced by apgParser#term.
    def exitTerm(self, ctx:apgParser.TermContext):
        pass


    # Enter a parse tree produced by apgParser#factor.
    def enterFactor(self, ctx:apgParser.FactorContext):
        pass

    # Exit a parse tree produced by apgParser#factor.
    def exitFactor(self, ctx:apgParser.FactorContext):
        pass


    # Enter a parse tree produced by apgParser#power.
    def enterPower(self, ctx:apgParser.PowerContext):
        pass

    # Exit a parse tree produced by apgParser#power.
    def exitPower(self, ctx:apgParser.PowerContext):
        pass


    # Enter a parse tree produced by apgParser#atom_expr.
    def enterAtom_expr(self, ctx:apgParser.Atom_exprContext):
        pass

    # Exit a parse tree produced by apgParser#atom_expr.
    def exitAtom_expr(self, ctx:apgParser.Atom_exprContext):
        pass


    # Enter a parse tree produced by apgParser#atom.
    def enterAtom(self, ctx:apgParser.AtomContext):
        pass

    # Exit a parse tree produced by apgParser#atom.
    def exitAtom(self, ctx:apgParser.AtomContext):
        pass


    # Enter a parse tree produced by apgParser#await_expr.
    def enterAwait_expr(self, ctx:apgParser.Await_exprContext):
        pass

    # Exit a parse tree produced by apgParser#await_expr.
    def exitAwait_expr(self, ctx:apgParser.Await_exprContext):
        pass


    # Enter a parse tree produced by apgParser#trailer.
    def enterTrailer(self, ctx:apgParser.TrailerContext):
        pass

    # Exit a parse tree produced by apgParser#trailer.
    def exitTrailer(self, ctx:apgParser.TrailerContext):
        pass


    # Enter a parse tree produced by apgParser#subscriptlist.
    def enterSubscriptlist(self, ctx:apgParser.SubscriptlistContext):
        pass

    # Exit a parse tree produced by apgParser#subscriptlist.
    def exitSubscriptlist(self, ctx:apgParser.SubscriptlistContext):
        pass


    # Enter a parse tree produced by apgParser#subscript.
    def enterSubscript(self, ctx:apgParser.SubscriptContext):
        pass

    # Exit a parse tree produced by apgParser#subscript.
    def exitSubscript(self, ctx:apgParser.SubscriptContext):
        pass


    # Enter a parse tree produced by apgParser#sliceop.
    def enterSliceop(self, ctx:apgParser.SliceopContext):
        pass

    # Exit a parse tree produced by apgParser#sliceop.
    def exitSliceop(self, ctx:apgParser.SliceopContext):
        pass


    # Enter a parse tree produced by apgParser#listmaker.
    def enterListmaker(self, ctx:apgParser.ListmakerContext):
        pass

    # Exit a parse tree produced by apgParser#listmaker.
    def exitListmaker(self, ctx:apgParser.ListmakerContext):
        pass


    # Enter a parse tree produced by apgParser#dictorsetmaker.
    def enterDictorsetmaker(self, ctx:apgParser.DictorsetmakerContext):
        pass

    # Exit a parse tree produced by apgParser#dictorsetmaker.
    def exitDictorsetmaker(self, ctx:apgParser.DictorsetmakerContext):
        pass


    # Enter a parse tree produced by apgParser#testlist_comp.
    def enterTestlist_comp(self, ctx:apgParser.Testlist_compContext):
        pass

    # Exit a parse tree produced by apgParser#testlist_comp.
    def exitTestlist_comp(self, ctx:apgParser.Testlist_compContext):
        pass


    # Enter a parse tree produced by apgParser#star_expr.
    def enterStar_expr(self, ctx:apgParser.Star_exprContext):
        pass

    # Exit a parse tree produced by apgParser#star_expr.
    def exitStar_expr(self, ctx:apgParser.Star_exprContext):
        pass


    # Enter a parse tree produced by apgParser#comp_for.
    def enterComp_for(self, ctx:apgParser.Comp_forContext):
        pass

    # Exit a parse tree produced by apgParser#comp_for.
    def exitComp_for(self, ctx:apgParser.Comp_forContext):
        pass


    # Enter a parse tree produced by apgParser#list_for.
    def enterList_for(self, ctx:apgParser.List_forContext):
        pass

    # Exit a parse tree produced by apgParser#list_for.
    def exitList_for(self, ctx:apgParser.List_forContext):
        pass


    # Enter a parse tree produced by apgParser#exprlist.
    def enterExprlist(self, ctx:apgParser.ExprlistContext):
        pass

    # Exit a parse tree produced by apgParser#exprlist.
    def exitExprlist(self, ctx:apgParser.ExprlistContext):
        pass


    # Enter a parse tree produced by apgParser#testlist.
    def enterTestlist(self, ctx:apgParser.TestlistContext):
        pass

    # Exit a parse tree produced by apgParser#testlist.
    def exitTestlist(self, ctx:apgParser.TestlistContext):
        pass


    # Enter a parse tree produced by apgParser#yield_expr.
    def enterYield_expr(self, ctx:apgParser.Yield_exprContext):
        pass

    # Exit a parse tree produced by apgParser#yield_expr.
    def exitYield_expr(self, ctx:apgParser.Yield_exprContext):
        pass


    # Enter a parse tree produced by apgParser#yield_arg.
    def enterYield_arg(self, ctx:apgParser.Yield_argContext):
        pass

    # Exit a parse tree produced by apgParser#yield_arg.
    def exitYield_arg(self, ctx:apgParser.Yield_argContext):
        pass


    # Enter a parse tree produced by apgParser#flow_definition.
    def enterFlow_definition(self, ctx:apgParser.Flow_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#flow_definition.
    def exitFlow_definition(self, ctx:apgParser.Flow_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#flow_step.
    def enterFlow_step(self, ctx:apgParser.Flow_stepContext):
        pass

    # Exit a parse tree produced by apgParser#flow_step.
    def exitFlow_step(self, ctx:apgParser.Flow_stepContext):
        pass


    # Enter a parse tree produced by apgParser#flow_connector.
    def enterFlow_connector(self, ctx:apgParser.Flow_connectorContext):
        pass

    # Exit a parse tree produced by apgParser#flow_connector.
    def exitFlow_connector(self, ctx:apgParser.Flow_connectorContext):
        pass


    # Enter a parse tree produced by apgParser#flow_modifiers.
    def enterFlow_modifiers(self, ctx:apgParser.Flow_modifiersContext):
        pass

    # Exit a parse tree produced by apgParser#flow_modifiers.
    def exitFlow_modifiers(self, ctx:apgParser.Flow_modifiersContext):
        pass


    # Enter a parse tree produced by apgParser#flow_modifier.
    def enterFlow_modifier(self, ctx:apgParser.Flow_modifierContext):
        pass

    # Exit a parse tree produced by apgParser#flow_modifier.
    def exitFlow_modifier(self, ctx:apgParser.Flow_modifierContext):
        pass


    # Enter a parse tree produced by apgParser#conditional_flow_step.
    def enterConditional_flow_step(self, ctx:apgParser.Conditional_flow_stepContext):
        pass

    # Exit a parse tree produced by apgParser#conditional_flow_step.
    def exitConditional_flow_step(self, ctx:apgParser.Conditional_flow_stepContext):
        pass


    # Enter a parse tree produced by apgParser#parallel_flow_step.
    def enterParallel_flow_step(self, ctx:apgParser.Parallel_flow_stepContext):
        pass

    # Exit a parse tree produced by apgParser#parallel_flow_step.
    def exitParallel_flow_step(self, ctx:apgParser.Parallel_flow_stepContext):
        pass


    # Enter a parse tree produced by apgParser#minion_command.
    def enterMinion_command(self, ctx:apgParser.Minion_commandContext):
        pass

    # Exit a parse tree produced by apgParser#minion_command.
    def exitMinion_command(self, ctx:apgParser.Minion_commandContext):
        pass


    # Enter a parse tree produced by apgParser#broadcast_command.
    def enterBroadcast_command(self, ctx:apgParser.Broadcast_commandContext):
        pass

    # Exit a parse tree produced by apgParser#broadcast_command.
    def exitBroadcast_command(self, ctx:apgParser.Broadcast_commandContext):
        pass


    # Enter a parse tree produced by apgParser#minion_verb.
    def enterMinion_verb(self, ctx:apgParser.Minion_verbContext):
        pass

    # Exit a parse tree produced by apgParser#minion_verb.
    def exitMinion_verb(self, ctx:apgParser.Minion_verbContext):
        pass


    # Enter a parse tree produced by apgParser#minion_scope.
    def enterMinion_scope(self, ctx:apgParser.Minion_scopeContext):
        pass

    # Exit a parse tree produced by apgParser#minion_scope.
    def exitMinion_scope(self, ctx:apgParser.Minion_scopeContext):
        pass


    # Enter a parse tree produced by apgParser#nested_entity.
    def enterNested_entity(self, ctx:apgParser.Nested_entityContext):
        pass

    # Exit a parse tree produced by apgParser#nested_entity.
    def exitNested_entity(self, ctx:apgParser.Nested_entityContext):
        pass


    # Enter a parse tree produced by apgParser#class_def.
    def enterClass_def(self, ctx:apgParser.Class_defContext):
        pass

    # Exit a parse tree produced by apgParser#class_def.
    def exitClass_def(self, ctx:apgParser.Class_defContext):
        pass


    # Enter a parse tree produced by apgParser#class_body.
    def enterClass_body(self, ctx:apgParser.Class_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#class_body.
    def exitClass_body(self, ctx:apgParser.Class_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#class_member.
    def enterClass_member(self, ctx:apgParser.Class_memberContext):
        pass

    # Exit a parse tree produced by apgParser#class_member.
    def exitClass_member(self, ctx:apgParser.Class_memberContext):
        pass


    # Enter a parse tree produced by apgParser#exception_def.
    def enterException_def(self, ctx:apgParser.Exception_defContext):
        pass

    # Exit a parse tree produced by apgParser#exception_def.
    def exitException_def(self, ctx:apgParser.Exception_defContext):
        pass


    # Enter a parse tree produced by apgParser#exception_body.
    def enterException_body(self, ctx:apgParser.Exception_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#exception_body.
    def exitException_body(self, ctx:apgParser.Exception_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#variable_declaration.
    def enterVariable_declaration(self, ctx:apgParser.Variable_declarationContext):
        pass

    # Exit a parse tree produced by apgParser#variable_declaration.
    def exitVariable_declaration(self, ctx:apgParser.Variable_declarationContext):
        pass


    # Enter a parse tree produced by apgParser#ab_testing_config.
    def enterAb_testing_config(self, ctx:apgParser.Ab_testing_configContext):
        pass

    # Exit a parse tree produced by apgParser#ab_testing_config.
    def exitAb_testing_config(self, ctx:apgParser.Ab_testing_configContext):
        pass


    # Enter a parse tree produced by apgParser#access_policy_specification.
    def enterAccess_policy_specification(self, ctx:apgParser.Access_policy_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#access_policy_specification.
    def exitAccess_policy_specification(self, ctx:apgParser.Access_policy_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#aggregation_specification.
    def enterAggregation_specification(self, ctx:apgParser.Aggregation_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#aggregation_specification.
    def exitAggregation_specification(self, ctx:apgParser.Aggregation_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#alert_frequency.
    def enterAlert_frequency(self, ctx:apgParser.Alert_frequencyContext):
        pass

    # Exit a parse tree produced by apgParser#alert_frequency.
    def exitAlert_frequency(self, ctx:apgParser.Alert_frequencyContext):
        pass


    # Enter a parse tree produced by apgParser#alerting_specification.
    def enterAlerting_specification(self, ctx:apgParser.Alerting_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#alerting_specification.
    def exitAlerting_specification(self, ctx:apgParser.Alerting_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#algorithm_specification.
    def enterAlgorithm_specification(self, ctx:apgParser.Algorithm_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#algorithm_specification.
    def exitAlgorithm_specification(self, ctx:apgParser.Algorithm_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#analytics_specification.
    def enterAnalytics_specification(self, ctx:apgParser.Analytics_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#analytics_specification.
    def exitAnalytics_specification(self, ctx:apgParser.Analytics_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#announcement_config.
    def enterAnnouncement_config(self, ctx:apgParser.Announcement_configContext):
        pass

    # Exit a parse tree produced by apgParser#announcement_config.
    def exitAnnouncement_config(self, ctx:apgParser.Announcement_configContext):
        pass


    # Enter a parse tree produced by apgParser#api_call.
    def enterApi_call(self, ctx:apgParser.Api_callContext):
        pass

    # Exit a parse tree produced by apgParser#api_call.
    def exitApi_call(self, ctx:apgParser.Api_callContext):
        pass


    # Enter a parse tree produced by apgParser#api_gateway_config.
    def enterApi_gateway_config(self, ctx:apgParser.Api_gateway_configContext):
        pass

    # Exit a parse tree produced by apgParser#api_gateway_config.
    def exitApi_gateway_config(self, ctx:apgParser.Api_gateway_configContext):
        pass


    # Enter a parse tree produced by apgParser#api_specification.
    def enterApi_specification(self, ctx:apgParser.Api_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#api_specification.
    def exitApi_specification(self, ctx:apgParser.Api_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#auto_resolution_specification.
    def enterAuto_resolution_specification(self, ctx:apgParser.Auto_resolution_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#auto_resolution_specification.
    def exitAuto_resolution_specification(self, ctx:apgParser.Auto_resolution_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#autocomplete_config.
    def enterAutocomplete_config(self, ctx:apgParser.Autocomplete_configContext):
        pass

    # Exit a parse tree produced by apgParser#autocomplete_config.
    def exitAutocomplete_config(self, ctx:apgParser.Autocomplete_configContext):
        pass


    # Enter a parse tree produced by apgParser#background_check_config.
    def enterBackground_check_config(self, ctx:apgParser.Background_check_configContext):
        pass

    # Exit a parse tree produced by apgParser#background_check_config.
    def exitBackground_check_config(self, ctx:apgParser.Background_check_configContext):
        pass


    # Enter a parse tree produced by apgParser#backup_specification.
    def enterBackup_specification(self, ctx:apgParser.Backup_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#backup_specification.
    def exitBackup_specification(self, ctx:apgParser.Backup_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#bi_configuration.
    def enterBi_configuration(self, ctx:apgParser.Bi_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#bi_configuration.
    def exitBi_configuration(self, ctx:apgParser.Bi_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#bucket_specification.
    def enterBucket_specification(self, ctx:apgParser.Bucket_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#bucket_specification.
    def exitBucket_specification(self, ctx:apgParser.Bucket_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#chat_configuration.
    def enterChat_configuration(self, ctx:apgParser.Chat_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#chat_configuration.
    def exitChat_configuration(self, ctx:apgParser.Chat_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#circuit_breaker_config.
    def enterCircuit_breaker_config(self, ctx:apgParser.Circuit_breaker_configContext):
        pass

    # Exit a parse tree produced by apgParser#circuit_breaker_config.
    def exitCircuit_breaker_config(self, ctx:apgParser.Circuit_breaker_configContext):
        pass


    # Enter a parse tree produced by apgParser#cohort_analysis_config.
    def enterCohort_analysis_config(self, ctx:apgParser.Cohort_analysis_configContext):
        pass

    # Exit a parse tree produced by apgParser#cohort_analysis_config.
    def exitCohort_analysis_config(self, ctx:apgParser.Cohort_analysis_configContext):
        pass


    # Enter a parse tree produced by apgParser#condition_specification.
    def enterCondition_specification(self, ctx:apgParser.Condition_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#condition_specification.
    def exitCondition_specification(self, ctx:apgParser.Condition_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#constraint_specification.
    def enterConstraint_specification(self, ctx:apgParser.Constraint_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#constraint_specification.
    def exitConstraint_specification(self, ctx:apgParser.Constraint_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#delivery_specification.
    def enterDelivery_specification(self, ctx:apgParser.Delivery_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#delivery_specification.
    def exitDelivery_specification(self, ctx:apgParser.Delivery_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#document_verification_config.
    def enterDocument_verification_config(self, ctx:apgParser.Document_verification_configContext):
        pass

    # Exit a parse tree produced by apgParser#document_verification_config.
    def exitDocument_verification_config(self, ctx:apgParser.Document_verification_configContext):
        pass


    # Enter a parse tree produced by apgParser#escalation_specification.
    def enterEscalation_specification(self, ctx:apgParser.Escalation_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#escalation_specification.
    def exitEscalation_specification(self, ctx:apgParser.Escalation_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#event_persistence_config.
    def enterEvent_persistence_config(self, ctx:apgParser.Event_persistence_configContext):
        pass

    # Exit a parse tree produced by apgParser#event_persistence_config.
    def exitEvent_persistence_config(self, ctx:apgParser.Event_persistence_configContext):
        pass


    # Enter a parse tree produced by apgParser#fraud_detection_config.
    def enterFraud_detection_config(self, ctx:apgParser.Fraud_detection_configContext):
        pass

    # Exit a parse tree produced by apgParser#fraud_detection_config.
    def exitFraud_detection_config(self, ctx:apgParser.Fraud_detection_configContext):
        pass


    # Enter a parse tree produced by apgParser#funnel_analysis_config.
    def enterFunnel_analysis_config(self, ctx:apgParser.Funnel_analysis_configContext):
        pass

    # Exit a parse tree produced by apgParser#funnel_analysis_config.
    def exitFunnel_analysis_config(self, ctx:apgParser.Funnel_analysis_configContext):
        pass


    # Enter a parse tree produced by apgParser#group_reference.
    def enterGroup_reference(self, ctx:apgParser.Group_referenceContext):
        pass

    # Exit a parse tree produced by apgParser#group_reference.
    def exitGroup_reference(self, ctx:apgParser.Group_referenceContext):
        pass


    # Enter a parse tree produced by apgParser#identity_verification_config.
    def enterIdentity_verification_config(self, ctx:apgParser.Identity_verification_configContext):
        pass

    # Exit a parse tree produced by apgParser#identity_verification_config.
    def exitIdentity_verification_config(self, ctx:apgParser.Identity_verification_configContext):
        pass


    # Enter a parse tree produced by apgParser#kyc_config.
    def enterKyc_config(self, ctx:apgParser.Kyc_configContext):
        pass

    # Exit a parse tree produced by apgParser#kyc_config.
    def exitKyc_config(self, ctx:apgParser.Kyc_configContext):
        pass


    # Enter a parse tree produced by apgParser#load_balancer_config.
    def enterLoad_balancer_config(self, ctx:apgParser.Load_balancer_configContext):
        pass

    # Exit a parse tree produced by apgParser#load_balancer_config.
    def exitLoad_balancer_config(self, ctx:apgParser.Load_balancer_configContext):
        pass


    # Enter a parse tree produced by apgParser#localization_specification.
    def enterLocalization_specification(self, ctx:apgParser.Localization_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#localization_specification.
    def exitLocalization_specification(self, ctx:apgParser.Localization_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#monitoring_specification.
    def enterMonitoring_specification(self, ctx:apgParser.Monitoring_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#monitoring_specification.
    def exitMonitoring_specification(self, ctx:apgParser.Monitoring_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#personalization_specification.
    def enterPersonalization_specification(self, ctx:apgParser.Personalization_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#personalization_specification.
    def exitPersonalization_specification(self, ctx:apgParser.Personalization_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#push_notification_config.
    def enterPush_notification_config(self, ctx:apgParser.Push_notification_configContext):
        pass

    # Exit a parse tree produced by apgParser#push_notification_config.
    def exitPush_notification_config(self, ctx:apgParser.Push_notification_configContext):
        pass


    # Enter a parse tree produced by apgParser#rate_limit_specification.
    def enterRate_limit_specification(self, ctx:apgParser.Rate_limit_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#rate_limit_specification.
    def exitRate_limit_specification(self, ctx:apgParser.Rate_limit_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#retry_policy_config.
    def enterRetry_policy_config(self, ctx:apgParser.Retry_policy_configContext):
        pass

    # Exit a parse tree produced by apgParser#retry_policy_config.
    def exitRetry_policy_config(self, ctx:apgParser.Retry_policy_configContext):
        pass


    # Enter a parse tree produced by apgParser#retry_specification.
    def enterRetry_specification(self, ctx:apgParser.Retry_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#retry_specification.
    def exitRetry_specification(self, ctx:apgParser.Retry_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#role_reference.
    def enterRole_reference(self, ctx:apgParser.Role_referenceContext):
        pass

    # Exit a parse tree produced by apgParser#role_reference.
    def exitRole_reference(self, ctx:apgParser.Role_referenceContext):
        pass


    # Enter a parse tree produced by apgParser#schedule_specification.
    def enterSchedule_specification(self, ctx:apgParser.Schedule_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#schedule_specification.
    def exitSchedule_specification(self, ctx:apgParser.Schedule_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#schema_specification.
    def enterSchema_specification(self, ctx:apgParser.Schema_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#schema_specification.
    def exitSchema_specification(self, ctx:apgParser.Schema_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#service_mesh_config.
    def enterService_mesh_config(self, ctx:apgParser.Service_mesh_configContext):
        pass

    # Exit a parse tree produced by apgParser#service_mesh_config.
    def exitService_mesh_config(self, ctx:apgParser.Service_mesh_configContext):
        pass


    # Enter a parse tree produced by apgParser#sms_config.
    def enterSms_config(self, ctx:apgParser.Sms_configContext):
        pass

    # Exit a parse tree produced by apgParser#sms_config.
    def exitSms_config(self, ctx:apgParser.Sms_configContext):
        pass


    # Enter a parse tree produced by apgParser#template_specification.
    def enterTemplate_specification(self, ctx:apgParser.Template_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#template_specification.
    def exitTemplate_specification(self, ctx:apgParser.Template_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#tracking_specification.
    def enterTracking_specification(self, ctx:apgParser.Tracking_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#tracking_specification.
    def exitTracking_specification(self, ctx:apgParser.Tracking_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#transformation_specification.
    def enterTransformation_specification(self, ctx:apgParser.Transformation_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#transformation_specification.
    def exitTransformation_specification(self, ctx:apgParser.Transformation_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#user_reference.
    def enterUser_reference(self, ctx:apgParser.User_referenceContext):
        pass

    # Exit a parse tree produced by apgParser#user_reference.
    def exitUser_reference(self, ctx:apgParser.User_referenceContext):
        pass


    # Enter a parse tree produced by apgParser#user_segmentation_config.
    def enterUser_segmentation_config(self, ctx:apgParser.User_segmentation_configContext):
        pass

    # Exit a parse tree produced by apgParser#user_segmentation_config.
    def exitUser_segmentation_config(self, ctx:apgParser.User_segmentation_configContext):
        pass


    # Enter a parse tree produced by apgParser#config_property.
    def enterConfig_property(self, ctx:apgParser.Config_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#config_property.
    def exitConfig_property(self, ctx:apgParser.Config_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#policy_property.
    def enterPolicy_property(self, ctx:apgParser.Policy_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#policy_property.
    def exitPolicy_property(self, ctx:apgParser.Policy_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#agg_property.
    def enterAgg_property(self, ctx:apgParser.Agg_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#agg_property.
    def exitAgg_property(self, ctx:apgParser.Agg_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#algo_property.
    def enterAlgo_property(self, ctx:apgParser.Algo_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#algo_property.
    def exitAlgo_property(self, ctx:apgParser.Algo_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#gateway_property.
    def enterGateway_property(self, ctx:apgParser.Gateway_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#gateway_property.
    def exitGateway_property(self, ctx:apgParser.Gateway_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#api_property.
    def enterApi_property(self, ctx:apgParser.Api_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#api_property.
    def exitApi_property(self, ctx:apgParser.Api_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#resolution_property.
    def enterResolution_property(self, ctx:apgParser.Resolution_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#resolution_property.
    def exitResolution_property(self, ctx:apgParser.Resolution_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#check_property.
    def enterCheck_property(self, ctx:apgParser.Check_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#check_property.
    def exitCheck_property(self, ctx:apgParser.Check_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#backup_property.
    def enterBackup_property(self, ctx:apgParser.Backup_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#backup_property.
    def exitBackup_property(self, ctx:apgParser.Backup_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#bi_property.
    def enterBi_property(self, ctx:apgParser.Bi_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#bi_property.
    def exitBi_property(self, ctx:apgParser.Bi_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#bucket_property.
    def enterBucket_property(self, ctx:apgParser.Bucket_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#bucket_property.
    def exitBucket_property(self, ctx:apgParser.Bucket_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#chat_property.
    def enterChat_property(self, ctx:apgParser.Chat_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#chat_property.
    def exitChat_property(self, ctx:apgParser.Chat_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#breaker_property.
    def enterBreaker_property(self, ctx:apgParser.Breaker_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#breaker_property.
    def exitBreaker_property(self, ctx:apgParser.Breaker_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#cohort_property.
    def enterCohort_property(self, ctx:apgParser.Cohort_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#cohort_property.
    def exitCohort_property(self, ctx:apgParser.Cohort_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#constraint_property.
    def enterConstraint_property(self, ctx:apgParser.Constraint_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#constraint_property.
    def exitConstraint_property(self, ctx:apgParser.Constraint_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#delivery_property.
    def enterDelivery_property(self, ctx:apgParser.Delivery_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#delivery_property.
    def exitDelivery_property(self, ctx:apgParser.Delivery_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#doc_property.
    def enterDoc_property(self, ctx:apgParser.Doc_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#doc_property.
    def exitDoc_property(self, ctx:apgParser.Doc_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#escalation_property.
    def enterEscalation_property(self, ctx:apgParser.Escalation_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#escalation_property.
    def exitEscalation_property(self, ctx:apgParser.Escalation_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#persistence_property.
    def enterPersistence_property(self, ctx:apgParser.Persistence_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#persistence_property.
    def exitPersistence_property(self, ctx:apgParser.Persistence_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#fraud_property.
    def enterFraud_property(self, ctx:apgParser.Fraud_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#fraud_property.
    def exitFraud_property(self, ctx:apgParser.Fraud_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#funnel_property.
    def enterFunnel_property(self, ctx:apgParser.Funnel_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#funnel_property.
    def exitFunnel_property(self, ctx:apgParser.Funnel_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#identity_property.
    def enterIdentity_property(self, ctx:apgParser.Identity_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#identity_property.
    def exitIdentity_property(self, ctx:apgParser.Identity_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#kyc_property.
    def enterKyc_property(self, ctx:apgParser.Kyc_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#kyc_property.
    def exitKyc_property(self, ctx:apgParser.Kyc_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#lb_property.
    def enterLb_property(self, ctx:apgParser.Lb_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#lb_property.
    def exitLb_property(self, ctx:apgParser.Lb_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#localization_property.
    def enterLocalization_property(self, ctx:apgParser.Localization_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#localization_property.
    def exitLocalization_property(self, ctx:apgParser.Localization_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#monitoring_property.
    def enterMonitoring_property(self, ctx:apgParser.Monitoring_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#monitoring_property.
    def exitMonitoring_property(self, ctx:apgParser.Monitoring_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#personalization_property.
    def enterPersonalization_property(self, ctx:apgParser.Personalization_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#personalization_property.
    def exitPersonalization_property(self, ctx:apgParser.Personalization_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#push_property.
    def enterPush_property(self, ctx:apgParser.Push_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#push_property.
    def exitPush_property(self, ctx:apgParser.Push_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#rate_limit_property.
    def enterRate_limit_property(self, ctx:apgParser.Rate_limit_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#rate_limit_property.
    def exitRate_limit_property(self, ctx:apgParser.Rate_limit_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#retry_property.
    def enterRetry_property(self, ctx:apgParser.Retry_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#retry_property.
    def exitRetry_property(self, ctx:apgParser.Retry_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#schedule_property.
    def enterSchedule_property(self, ctx:apgParser.Schedule_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#schedule_property.
    def exitSchedule_property(self, ctx:apgParser.Schedule_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#template_property.
    def enterTemplate_property(self, ctx:apgParser.Template_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#template_property.
    def exitTemplate_property(self, ctx:apgParser.Template_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#tracking_property.
    def enterTracking_property(self, ctx:apgParser.Tracking_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#tracking_property.
    def exitTracking_property(self, ctx:apgParser.Tracking_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#transform_property.
    def enterTransform_property(self, ctx:apgParser.Transform_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#transform_property.
    def exitTransform_property(self, ctx:apgParser.Transform_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#segment_property.
    def enterSegment_property(self, ctx:apgParser.Segment_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#segment_property.
    def exitSegment_property(self, ctx:apgParser.Segment_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#mesh_property.
    def enterMesh_property(self, ctx:apgParser.Mesh_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#mesh_property.
    def exitMesh_property(self, ctx:apgParser.Mesh_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#sms_property.
    def enterSms_property(self, ctx:apgParser.Sms_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#sms_property.
    def exitSms_property(self, ctx:apgParser.Sms_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#commission_config.
    def enterCommission_config(self, ctx:apgParser.Commission_configContext):
        pass

    # Exit a parse tree produced by apgParser#commission_config.
    def exitCommission_config(self, ctx:apgParser.Commission_configContext):
        pass


    # Enter a parse tree produced by apgParser#compliance_framework_list.
    def enterCompliance_framework_list(self, ctx:apgParser.Compliance_framework_listContext):
        pass

    # Exit a parse tree produced by apgParser#compliance_framework_list.
    def exitCompliance_framework_list(self, ctx:apgParser.Compliance_framework_listContext):
        pass


    # Enter a parse tree produced by apgParser#composite_condition.
    def enterComposite_condition(self, ctx:apgParser.Composite_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#composite_condition.
    def exitComposite_condition(self, ctx:apgParser.Composite_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#compression_specification.
    def enterCompression_specification(self, ctx:apgParser.Compression_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#compression_specification.
    def exitCompression_specification(self, ctx:apgParser.Compression_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#container_configuration.
    def enterContainer_configuration(self, ctx:apgParser.Container_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#container_configuration.
    def exitContainer_configuration(self, ctx:apgParser.Container_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#conversion_tracking_config.
    def enterConversion_tracking_config(self, ctx:apgParser.Conversion_tracking_configContext):
        pass

    # Exit a parse tree produced by apgParser#conversion_tracking_config.
    def exitConversion_tracking_config(self, ctx:apgParser.Conversion_tracking_configContext):
        pass


    # Enter a parse tree produced by apgParser#correlation_specification.
    def enterCorrelation_specification(self, ctx:apgParser.Correlation_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#correlation_specification.
    def exitCorrelation_specification(self, ctx:apgParser.Correlation_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#currency_conversion_config.
    def enterCurrency_conversion_config(self, ctx:apgParser.Currency_conversion_configContext):
        pass

    # Exit a parse tree produced by apgParser#currency_conversion_config.
    def exitCurrency_conversion_config(self, ctx:apgParser.Currency_conversion_configContext):
        pass


    # Enter a parse tree produced by apgParser#currency_list.
    def enterCurrency_list(self, ctx:apgParser.Currency_listContext):
        pass

    # Exit a parse tree produced by apgParser#currency_list.
    def exitCurrency_list(self, ctx:apgParser.Currency_listContext):
        pass


    # Enter a parse tree produced by apgParser#custom_condition.
    def enterCustom_condition(self, ctx:apgParser.Custom_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#custom_condition.
    def exitCustom_condition(self, ctx:apgParser.Custom_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#custom_format_specification.
    def enterCustom_format_specification(self, ctx:apgParser.Custom_format_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#custom_format_specification.
    def exitCustom_format_specification(self, ctx:apgParser.Custom_format_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#custom_unit.
    def enterCustom_unit(self, ctx:apgParser.Custom_unitContext):
        pass

    # Exit a parse tree produced by apgParser#custom_unit.
    def exitCustom_unit(self, ctx:apgParser.Custom_unitContext):
        pass


    # Enter a parse tree produced by apgParser#dashboard_configuration.
    def enterDashboard_configuration(self, ctx:apgParser.Dashboard_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#dashboard_configuration.
    def exitDashboard_configuration(self, ctx:apgParser.Dashboard_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#data_access_config.
    def enterData_access_config(self, ctx:apgParser.Data_access_configContext):
        pass

    # Exit a parse tree produced by apgParser#data_access_config.
    def exitData_access_config(self, ctx:apgParser.Data_access_configContext):
        pass


    # Enter a parse tree produced by apgParser#data_warehouse_config.
    def enterData_warehouse_config(self, ctx:apgParser.Data_warehouse_configContext):
        pass

    # Exit a parse tree produced by apgParser#data_warehouse_config.
    def exitData_warehouse_config(self, ctx:apgParser.Data_warehouse_configContext):
        pass


    # Enter a parse tree produced by apgParser#database_reference.
    def enterDatabase_reference(self, ctx:apgParser.Database_referenceContext):
        pass

    # Exit a parse tree produced by apgParser#database_reference.
    def exitDatabase_reference(self, ctx:apgParser.Database_referenceContext):
        pass


    # Enter a parse tree produced by apgParser#deployment_environment.
    def enterDeployment_environment(self, ctx:apgParser.Deployment_environmentContext):
        pass

    # Exit a parse tree produced by apgParser#deployment_environment.
    def exitDeployment_environment(self, ctx:apgParser.Deployment_environmentContext):
        pass


    # Enter a parse tree produced by apgParser#dispute_config.
    def enterDispute_config(self, ctx:apgParser.Dispute_configContext):
        pass

    # Exit a parse tree produced by apgParser#dispute_config.
    def exitDispute_config(self, ctx:apgParser.Dispute_configContext):
        pass


    # Enter a parse tree produced by apgParser#duration_clause.
    def enterDuration_clause(self, ctx:apgParser.Duration_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#duration_clause.
    def exitDuration_clause(self, ctx:apgParser.Duration_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#ecommerce_config.
    def enterEcommerce_config(self, ctx:apgParser.Ecommerce_configContext):
        pass

    # Exit a parse tree produced by apgParser#ecommerce_config.
    def exitEcommerce_config(self, ctx:apgParser.Ecommerce_configContext):
        pass


    # Enter a parse tree produced by apgParser#environment_variable.
    def enterEnvironment_variable(self, ctx:apgParser.Environment_variableContext):
        pass

    # Exit a parse tree produced by apgParser#environment_variable.
    def exitEnvironment_variable(self, ctx:apgParser.Environment_variableContext):
        pass


    # Enter a parse tree produced by apgParser#error_handling_config.
    def enterError_handling_config(self, ctx:apgParser.Error_handling_configContext):
        pass

    # Exit a parse tree produced by apgParser#error_handling_config.
    def exitError_handling_config(self, ctx:apgParser.Error_handling_configContext):
        pass


    # Enter a parse tree produced by apgParser#escrow_config.
    def enterEscrow_config(self, ctx:apgParser.Escrow_configContext):
        pass

    # Exit a parse tree produced by apgParser#escrow_config.
    def exitEscrow_config(self, ctx:apgParser.Escrow_configContext):
        pass


    # Enter a parse tree produced by apgParser#execution_environment.
    def enterExecution_environment(self, ctx:apgParser.Execution_environmentContext):
        pass

    # Exit a parse tree produced by apgParser#execution_environment.
    def exitExecution_environment(self, ctx:apgParser.Execution_environmentContext):
        pass


    # Enter a parse tree produced by apgParser#experiment_configuration.
    def enterExperiment_configuration(self, ctx:apgParser.Experiment_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#experiment_configuration.
    def exitExperiment_configuration(self, ctx:apgParser.Experiment_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#failover_config.
    def enterFailover_config(self, ctx:apgParser.Failover_configContext):
        pass

    # Exit a parse tree produced by apgParser#failover_config.
    def exitFailover_config(self, ctx:apgParser.Failover_configContext):
        pass


    # Enter a parse tree produced by apgParser#feature_flag_config.
    def enterFeature_flag_config(self, ctx:apgParser.Feature_flag_configContext):
        pass

    # Exit a parse tree produced by apgParser#feature_flag_config.
    def exitFeature_flag_config(self, ctx:apgParser.Feature_flag_configContext):
        pass


    # Enter a parse tree produced by apgParser#fulfillment_config.
    def enterFulfillment_config(self, ctx:apgParser.Fulfillment_configContext):
        pass

    # Exit a parse tree produced by apgParser#fulfillment_config.
    def exitFulfillment_config(self, ctx:apgParser.Fulfillment_configContext):
        pass


    # Enter a parse tree produced by apgParser#gdpr_config.
    def enterGdpr_config(self, ctx:apgParser.Gdpr_configContext):
        pass

    # Exit a parse tree produced by apgParser#gdpr_config.
    def exitGdpr_config(self, ctx:apgParser.Gdpr_configContext):
        pass


    # Enter a parse tree produced by apgParser#health_check_config.
    def enterHealth_check_config(self, ctx:apgParser.Health_check_configContext):
        pass

    # Exit a parse tree produced by apgParser#health_check_config.
    def exitHealth_check_config(self, ctx:apgParser.Health_check_configContext):
        pass


    # Enter a parse tree produced by apgParser#identity_provider_config.
    def enterIdentity_provider_config(self, ctx:apgParser.Identity_provider_configContext):
        pass

    # Exit a parse tree produced by apgParser#identity_provider_config.
    def exitIdentity_provider_config(self, ctx:apgParser.Identity_provider_configContext):
        pass


    # Enter a parse tree produced by apgParser#infrastructure_requirement.
    def enterInfrastructure_requirement(self, ctx:apgParser.Infrastructure_requirementContext):
        pass

    # Exit a parse tree produced by apgParser#infrastructure_requirement.
    def exitInfrastructure_requirement(self, ctx:apgParser.Infrastructure_requirementContext):
        pass


    # Enter a parse tree produced by apgParser#inventory_config.
    def enterInventory_config(self, ctx:apgParser.Inventory_configContext):
        pass

    # Exit a parse tree produced by apgParser#inventory_config.
    def exitInventory_config(self, ctx:apgParser.Inventory_configContext):
        pass


    # Enter a parse tree produced by apgParser#lambda_configuration.
    def enterLambda_configuration(self, ctx:apgParser.Lambda_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#lambda_configuration.
    def exitLambda_configuration(self, ctx:apgParser.Lambda_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#listing_config.
    def enterListing_config(self, ctx:apgParser.Listing_configContext):
        pass

    # Exit a parse tree produced by apgParser#listing_config.
    def exitListing_config(self, ctx:apgParser.Listing_configContext):
        pass


    # Enter a parse tree produced by apgParser#machine_learning_config.
    def enterMachine_learning_config(self, ctx:apgParser.Machine_learning_configContext):
        pass

    # Exit a parse tree produced by apgParser#machine_learning_config.
    def exitMachine_learning_config(self, ctx:apgParser.Machine_learning_configContext):
        pass


    # Enter a parse tree produced by apgParser#ml_model_configuration.
    def enterMl_model_configuration(self, ctx:apgParser.Ml_model_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#ml_model_configuration.
    def exitMl_model_configuration(self, ctx:apgParser.Ml_model_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#model_configuration.
    def enterModel_configuration(self, ctx:apgParser.Model_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#model_configuration.
    def exitModel_configuration(self, ctx:apgParser.Model_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#notification_template.
    def enterNotification_template(self, ctx:apgParser.Notification_templateContext):
        pass

    # Exit a parse tree produced by apgParser#notification_template.
    def exitNotification_template(self, ctx:apgParser.Notification_templateContext):
        pass


    # Enter a parse tree produced by apgParser#orchestration_config.
    def enterOrchestration_config(self, ctx:apgParser.Orchestration_configContext):
        pass

    # Exit a parse tree produced by apgParser#orchestration_config.
    def exitOrchestration_config(self, ctx:apgParser.Orchestration_configContext):
        pass


    # Enter a parse tree produced by apgParser#payment_config.
    def enterPayment_config(self, ctx:apgParser.Payment_configContext):
        pass

    # Exit a parse tree produced by apgParser#payment_config.
    def exitPayment_config(self, ctx:apgParser.Payment_configContext):
        pass


    # Enter a parse tree produced by apgParser#platform_config.
    def enterPlatform_config(self, ctx:apgParser.Platform_configContext):
        pass

    # Exit a parse tree produced by apgParser#platform_config.
    def exitPlatform_config(self, ctx:apgParser.Platform_configContext):
        pass


    # Enter a parse tree produced by apgParser#platform_name.
    def enterPlatform_name(self, ctx:apgParser.Platform_nameContext):
        pass

    # Exit a parse tree produced by apgParser#platform_name.
    def exitPlatform_name(self, ctx:apgParser.Platform_nameContext):
        pass


    # Enter a parse tree produced by apgParser#prediction_configuration.
    def enterPrediction_configuration(self, ctx:apgParser.Prediction_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#prediction_configuration.
    def exitPrediction_configuration(self, ctx:apgParser.Prediction_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#pricing_config.
    def enterPricing_config(self, ctx:apgParser.Pricing_configContext):
        pass

    # Exit a parse tree produced by apgParser#pricing_config.
    def exitPricing_config(self, ctx:apgParser.Pricing_configContext):
        pass


    # Enter a parse tree produced by apgParser#quality_gate_config.
    def enterQuality_gate_config(self, ctx:apgParser.Quality_gate_configContext):
        pass

    # Exit a parse tree produced by apgParser#quality_gate_config.
    def exitQuality_gate_config(self, ctx:apgParser.Quality_gate_configContext):
        pass


    # Enter a parse tree produced by apgParser#region_list.
    def enterRegion_list(self, ctx:apgParser.Region_listContext):
        pass

    # Exit a parse tree produced by apgParser#region_list.
    def exitRegion_list(self, ctx:apgParser.Region_listContext):
        pass


    # Enter a parse tree produced by apgParser#resource_requirement.
    def enterResource_requirement(self, ctx:apgParser.Resource_requirementContext):
        pass

    # Exit a parse tree produced by apgParser#resource_requirement.
    def exitResource_requirement(self, ctx:apgParser.Resource_requirementContext):
        pass


    # Enter a parse tree produced by apgParser#retention_analysis_config.
    def enterRetention_analysis_config(self, ctx:apgParser.Retention_analysis_configContext):
        pass

    # Exit a parse tree produced by apgParser#retention_analysis_config.
    def exitRetention_analysis_config(self, ctx:apgParser.Retention_analysis_configContext):
        pass


    # Enter a parse tree produced by apgParser#revenue_sharing_config.
    def enterRevenue_sharing_config(self, ctx:apgParser.Revenue_sharing_configContext):
        pass

    # Exit a parse tree produced by apgParser#revenue_sharing_config.
    def exitRevenue_sharing_config(self, ctx:apgParser.Revenue_sharing_configContext):
        pass


    # Enter a parse tree produced by apgParser#rollback_config.
    def enterRollback_config(self, ctx:apgParser.Rollback_configContext):
        pass

    # Exit a parse tree produced by apgParser#rollback_config.
    def exitRollback_config(self, ctx:apgParser.Rollback_configContext):
        pass


    # Enter a parse tree produced by apgParser#scaling_config.
    def enterScaling_config(self, ctx:apgParser.Scaling_configContext):
        pass

    # Exit a parse tree produced by apgParser#scaling_config.
    def exitScaling_config(self, ctx:apgParser.Scaling_configContext):
        pass


    # Enter a parse tree produced by apgParser#security_policy.
    def enterSecurity_policy(self, ctx:apgParser.Security_policyContext):
        pass

    # Exit a parse tree produced by apgParser#security_policy.
    def exitSecurity_policy(self, ctx:apgParser.Security_policyContext):
        pass


    # Enter a parse tree produced by apgParser#subscription_config.
    def enterSubscription_config(self, ctx:apgParser.Subscription_configContext):
        pass

    # Exit a parse tree produced by apgParser#subscription_config.
    def exitSubscription_config(self, ctx:apgParser.Subscription_configContext):
        pass


    # Enter a parse tree produced by apgParser#tax_calculation_config.
    def enterTax_calculation_config(self, ctx:apgParser.Tax_calculation_configContext):
        pass

    # Exit a parse tree produced by apgParser#tax_calculation_config.
    def exitTax_calculation_config(self, ctx:apgParser.Tax_calculation_configContext):
        pass


    # Enter a parse tree produced by apgParser#time_range.
    def enterTime_range(self, ctx:apgParser.Time_rangeContext):
        pass

    # Exit a parse tree produced by apgParser#time_range.
    def exitTime_range(self, ctx:apgParser.Time_rangeContext):
        pass


    # Enter a parse tree produced by apgParser#user_type_definition.
    def enterUser_type_definition(self, ctx:apgParser.User_type_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#user_type_definition.
    def exitUser_type_definition(self, ctx:apgParser.User_type_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#webhook_config.
    def enterWebhook_config(self, ctx:apgParser.Webhook_configContext):
        pass

    # Exit a parse tree produced by apgParser#webhook_config.
    def exitWebhook_config(self, ctx:apgParser.Webhook_configContext):
        pass


    # Enter a parse tree produced by apgParser#ecommerce_name.
    def enterEcommerce_name(self, ctx:apgParser.Ecommerce_nameContext):
        pass

    # Exit a parse tree produced by apgParser#ecommerce_name.
    def exitEcommerce_name(self, ctx:apgParser.Ecommerce_nameContext):
        pass


    # Enter a parse tree produced by apgParser#encryption_specification.
    def enterEncryption_specification(self, ctx:apgParser.Encryption_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#encryption_specification.
    def exitEncryption_specification(self, ctx:apgParser.Encryption_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#endpoint_definition_list.
    def enterEndpoint_definition_list(self, ctx:apgParser.Endpoint_definition_listContext):
        pass

    # Exit a parse tree produced by apgParser#endpoint_definition_list.
    def exitEndpoint_definition_list(self, ctx:apgParser.Endpoint_definition_listContext):
        pass


    # Enter a parse tree produced by apgParser#enrichment_specification.
    def enterEnrichment_specification(self, ctx:apgParser.Enrichment_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#enrichment_specification.
    def exitEnrichment_specification(self, ctx:apgParser.Enrichment_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#escalation_policy_specification.
    def enterEscalation_policy_specification(self, ctx:apgParser.Escalation_policy_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#escalation_policy_specification.
    def exitEscalation_policy_specification(self, ctx:apgParser.Escalation_policy_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#event_name.
    def enterEvent_name(self, ctx:apgParser.Event_nameContext):
        pass

    # Exit a parse tree produced by apgParser#event_name.
    def exitEvent_name(self, ctx:apgParser.Event_nameContext):
        pass


    # Enter a parse tree produced by apgParser#event_routing_config.
    def enterEvent_routing_config(self, ctx:apgParser.Event_routing_configContext):
        pass

    # Exit a parse tree produced by apgParser#event_routing_config.
    def exitEvent_routing_config(self, ctx:apgParser.Event_routing_configContext):
        pass


    # Enter a parse tree produced by apgParser#event_schema.
    def enterEvent_schema(self, ctx:apgParser.Event_schemaContext):
        pass

    # Exit a parse tree produced by apgParser#event_schema.
    def exitEvent_schema(self, ctx:apgParser.Event_schemaContext):
        pass


    # Enter a parse tree produced by apgParser#export_specification.
    def enterExport_specification(self, ctx:apgParser.Export_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#export_specification.
    def exitExport_specification(self, ctx:apgParser.Export_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#facet_configuration.
    def enterFacet_configuration(self, ctx:apgParser.Facet_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#facet_configuration.
    def exitFacet_configuration(self, ctx:apgParser.Facet_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#fee_configuration.
    def enterFee_configuration(self, ctx:apgParser.Fee_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#fee_configuration.
    def exitFee_configuration(self, ctx:apgParser.Fee_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#filter_specification.
    def enterFilter_specification(self, ctx:apgParser.Filter_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#filter_specification.
    def exitFilter_specification(self, ctx:apgParser.Filter_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#fraud_prevention_config.
    def enterFraud_prevention_config(self, ctx:apgParser.Fraud_prevention_configContext):
        pass

    # Exit a parse tree produced by apgParser#fraud_prevention_config.
    def exitFraud_prevention_config(self, ctx:apgParser.Fraud_prevention_configContext):
        pass


    # Enter a parse tree produced by apgParser#geolocation_config.
    def enterGeolocation_config(self, ctx:apgParser.Geolocation_configContext):
        pass

    # Exit a parse tree produced by apgParser#geolocation_config.
    def exitGeolocation_config(self, ctx:apgParser.Geolocation_configContext):
        pass


    # Enter a parse tree produced by apgParser#handler_list.
    def enterHandler_list(self, ctx:apgParser.Handler_listContext):
        pass

    # Exit a parse tree produced by apgParser#handler_list.
    def exitHandler_list(self, ctx:apgParser.Handler_listContext):
        pass


    # Enter a parse tree produced by apgParser#indexing_configuration.
    def enterIndexing_configuration(self, ctx:apgParser.Indexing_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#indexing_configuration.
    def exitIndexing_configuration(self, ctx:apgParser.Indexing_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#inventory_management.
    def enterInventory_management(self, ctx:apgParser.Inventory_managementContext):
        pass

    # Exit a parse tree produced by apgParser#inventory_management.
    def exitInventory_management(self, ctx:apgParser.Inventory_managementContext):
        pass


    # Enter a parse tree produced by apgParser#label_specification.
    def enterLabel_specification(self, ctx:apgParser.Label_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#label_specification.
    def exitLabel_specification(self, ctx:apgParser.Label_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#load_balancing_config.
    def enterLoad_balancing_config(self, ctx:apgParser.Load_balancing_configContext):
        pass

    # Exit a parse tree produced by apgParser#load_balancing_config.
    def exitLoad_balancing_config(self, ctx:apgParser.Load_balancing_configContext):
        pass


    # Enter a parse tree produced by apgParser#localization_config.
    def enterLocalization_config(self, ctx:apgParser.Localization_configContext):
        pass

    # Exit a parse tree produced by apgParser#localization_config.
    def exitLocalization_config(self, ctx:apgParser.Localization_configContext):
        pass


    # Enter a parse tree produced by apgParser#marketplace_name.
    def enterMarketplace_name(self, ctx:apgParser.Marketplace_nameContext):
        pass

    # Exit a parse tree produced by apgParser#marketplace_name.
    def exitMarketplace_name(self, ctx:apgParser.Marketplace_nameContext):
        pass


    # Enter a parse tree produced by apgParser#masking_specification.
    def enterMasking_specification(self, ctx:apgParser.Masking_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#masking_specification.
    def exitMasking_specification(self, ctx:apgParser.Masking_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#messaging_configuration.
    def enterMessaging_configuration(self, ctx:apgParser.Messaging_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#messaging_configuration.
    def exitMessaging_configuration(self, ctx:apgParser.Messaging_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#metadata_specification.
    def enterMetadata_specification(self, ctx:apgParser.Metadata_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#metadata_specification.
    def exitMetadata_specification(self, ctx:apgParser.Metadata_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#metric_reference.
    def enterMetric_reference(self, ctx:apgParser.Metric_referenceContext):
        pass

    # Exit a parse tree produced by apgParser#metric_reference.
    def exitMetric_reference(self, ctx:apgParser.Metric_referenceContext):
        pass


    # Enter a parse tree produced by apgParser#ml_analytics_config.
    def enterMl_analytics_config(self, ctx:apgParser.Ml_analytics_configContext):
        pass

    # Exit a parse tree produced by apgParser#ml_analytics_config.
    def exitMl_analytics_config(self, ctx:apgParser.Ml_analytics_configContext):
        pass


    # Enter a parse tree produced by apgParser#moderation_config.
    def enterModeration_config(self, ctx:apgParser.Moderation_configContext):
        pass

    # Exit a parse tree produced by apgParser#moderation_config.
    def exitModeration_config(self, ctx:apgParser.Moderation_configContext):
        pass


    # Enter a parse tree produced by apgParser#monitoring_configuration.
    def enterMonitoring_configuration(self, ctx:apgParser.Monitoring_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#monitoring_configuration.
    def exitMonitoring_configuration(self, ctx:apgParser.Monitoring_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#negotiation_config.
    def enterNegotiation_config(self, ctx:apgParser.Negotiation_configContext):
        pass

    # Exit a parse tree produced by apgParser#negotiation_config.
    def exitNegotiation_config(self, ctx:apgParser.Negotiation_configContext):
        pass


    # Enter a parse tree produced by apgParser#networking_configuration.
    def enterNetworking_configuration(self, ctx:apgParser.Networking_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#networking_configuration.
    def exitNetworking_configuration(self, ctx:apgParser.Networking_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#onboarding_definition.
    def enterOnboarding_definition(self, ctx:apgParser.Onboarding_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#onboarding_definition.
    def exitOnboarding_definition(self, ctx:apgParser.Onboarding_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#order_fulfillment.
    def enterOrder_fulfillment(self, ctx:apgParser.Order_fulfillmentContext):
        pass

    # Exit a parse tree produced by apgParser#order_fulfillment.
    def exitOrder_fulfillment(self, ctx:apgParser.Order_fulfillmentContext):
        pass


    # Enter a parse tree produced by apgParser#output_specification.
    def enterOutput_specification(self, ctx:apgParser.Output_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#output_specification.
    def exitOutput_specification(self, ctx:apgParser.Output_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#pattern_condition.
    def enterPattern_condition(self, ctx:apgParser.Pattern_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#pattern_condition.
    def exitPattern_condition(self, ctx:apgParser.Pattern_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#payment_method_list.
    def enterPayment_method_list(self, ctx:apgParser.Payment_method_listContext):
        pass

    # Exit a parse tree produced by apgParser#payment_method_list.
    def exitPayment_method_list(self, ctx:apgParser.Payment_method_listContext):
        pass


    # Enter a parse tree produced by apgParser#payment_provider_list.
    def enterPayment_provider_list(self, ctx:apgParser.Payment_provider_listContext):
        pass

    # Exit a parse tree produced by apgParser#payment_provider_list.
    def exitPayment_provider_list(self, ctx:apgParser.Payment_provider_listContext):
        pass


    # Enter a parse tree produced by apgParser#percentage_clause.
    def enterPercentage_clause(self, ctx:apgParser.Percentage_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#percentage_clause.
    def exitPercentage_clause(self, ctx:apgParser.Percentage_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#permission_list.
    def enterPermission_list(self, ctx:apgParser.Permission_listContext):
        pass

    # Exit a parse tree produced by apgParser#permission_list.
    def exitPermission_list(self, ctx:apgParser.Permission_listContext):
        pass


    # Enter a parse tree produced by apgParser#personalization_config.
    def enterPersonalization_config(self, ctx:apgParser.Personalization_configContext):
        pass

    # Exit a parse tree produced by apgParser#personalization_config.
    def exitPersonalization_config(self, ctx:apgParser.Personalization_configContext):
        pass


    # Enter a parse tree produced by apgParser#placement_strategy_config.
    def enterPlacement_strategy_config(self, ctx:apgParser.Placement_strategy_configContext):
        pass

    # Exit a parse tree produced by apgParser#placement_strategy_config.
    def exitPlacement_strategy_config(self, ctx:apgParser.Placement_strategy_configContext):
        pass


    # Enter a parse tree produced by apgParser#predictive_analytics_config.
    def enterPredictive_analytics_config(self, ctx:apgParser.Predictive_analytics_configContext):
        pass

    # Exit a parse tree produced by apgParser#predictive_analytics_config.
    def exitPredictive_analytics_config(self, ctx:apgParser.Predictive_analytics_configContext):
        pass


    # Enter a parse tree produced by apgParser#quantile_specification.
    def enterQuantile_specification(self, ctx:apgParser.Quantile_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#quantile_specification.
    def exitQuantile_specification(self, ctx:apgParser.Quantile_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#query_expression.
    def enterQuery_expression(self, ctx:apgParser.Query_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#query_expression.
    def exitQuery_expression(self, ctx:apgParser.Query_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#rating_configuration.
    def enterRating_configuration(self, ctx:apgParser.Rating_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#rating_configuration.
    def exitRating_configuration(self, ctx:apgParser.Rating_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#real_time_analytics_config.
    def enterReal_time_analytics_config(self, ctx:apgParser.Real_time_analytics_configContext):
        pass

    # Exit a parse tree produced by apgParser#real_time_analytics_config.
    def exitReal_time_analytics_config(self, ctx:apgParser.Real_time_analytics_configContext):
        pass


    # Enter a parse tree produced by apgParser#recommendation_config.
    def enterRecommendation_config(self, ctx:apgParser.Recommendation_configContext):
        pass

    # Exit a parse tree produced by apgParser#recommendation_config.
    def exitRecommendation_config(self, ctx:apgParser.Recommendation_configContext):
        pass


    # Enter a parse tree produced by apgParser#refund_configuration.
    def enterRefund_configuration(self, ctx:apgParser.Refund_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#refund_configuration.
    def exitRefund_configuration(self, ctx:apgParser.Refund_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#resource_requirements_config.
    def enterResource_requirements_config(self, ctx:apgParser.Resource_requirements_configContext):
        pass

    # Exit a parse tree produced by apgParser#resource_requirements_config.
    def exitResource_requirements_config(self, ctx:apgParser.Resource_requirements_configContext):
        pass


    # Enter a parse tree produced by apgParser#responsibility_list.
    def enterResponsibility_list(self, ctx:apgParser.Responsibility_listContext):
        pass

    # Exit a parse tree produced by apgParser#responsibility_list.
    def exitResponsibility_list(self, ctx:apgParser.Responsibility_listContext):
        pass


    # Enter a parse tree produced by apgParser#retention_specification.
    def enterRetention_specification(self, ctx:apgParser.Retention_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#retention_specification.
    def exitRetention_specification(self, ctx:apgParser.Retention_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#review_configuration.
    def enterReview_configuration(self, ctx:apgParser.Review_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#review_configuration.
    def exitReview_configuration(self, ctx:apgParser.Review_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#rotation_specification.
    def enterRotation_specification(self, ctx:apgParser.Rotation_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#rotation_specification.
    def exitRotation_specification(self, ctx:apgParser.Rotation_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#runbook_specification.
    def enterRunbook_specification(self, ctx:apgParser.Runbook_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#runbook_specification.
    def exitRunbook_specification(self, ctx:apgParser.Runbook_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#sampling_specification.
    def enterSampling_specification(self, ctx:apgParser.Sampling_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#sampling_specification.
    def exitSampling_specification(self, ctx:apgParser.Sampling_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#scaling_configuration.
    def enterScaling_configuration(self, ctx:apgParser.Scaling_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#scaling_configuration.
    def exitScaling_configuration(self, ctx:apgParser.Scaling_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#search_analytics_config.
    def enterSearch_analytics_config(self, ctx:apgParser.Search_analytics_configContext):
        pass

    # Exit a parse tree produced by apgParser#search_analytics_config.
    def exitSearch_analytics_config(self, ctx:apgParser.Search_analytics_configContext):
        pass


    # Enter a parse tree produced by apgParser#search_engine_type.
    def enterSearch_engine_type(self, ctx:apgParser.Search_engine_typeContext):
        pass

    # Exit a parse tree produced by apgParser#search_engine_type.
    def exitSearch_engine_type(self, ctx:apgParser.Search_engine_typeContext):
        pass


    # Enter a parse tree produced by apgParser#secrets_config.
    def enterSecrets_config(self, ctx:apgParser.Secrets_configContext):
        pass

    # Exit a parse tree produced by apgParser#secrets_config.
    def exitSecrets_config(self, ctx:apgParser.Secrets_configContext):
        pass


    # Enter a parse tree produced by apgParser#security_configuration.
    def enterSecurity_configuration(self, ctx:apgParser.Security_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#security_configuration.
    def exitSecurity_configuration(self, ctx:apgParser.Security_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#service_definition_list.
    def enterService_definition_list(self, ctx:apgParser.Service_definition_listContext):
        pass

    # Exit a parse tree produced by apgParser#service_definition_list.
    def exitService_definition_list(self, ctx:apgParser.Service_definition_listContext):
        pass


    # Enter a parse tree produced by apgParser#service_dependency_list.
    def enterService_dependency_list(self, ctx:apgParser.Service_dependency_listContext):
        pass

    # Exit a parse tree produced by apgParser#service_dependency_list.
    def exitService_dependency_list(self, ctx:apgParser.Service_dependency_listContext):
        pass


    # Enter a parse tree produced by apgParser#service_discovery_config.
    def enterService_discovery_config(self, ctx:apgParser.Service_discovery_configContext):
        pass

    # Exit a parse tree produced by apgParser#service_discovery_config.
    def exitService_discovery_config(self, ctx:apgParser.Service_discovery_configContext):
        pass


    # Enter a parse tree produced by apgParser#service_name.
    def enterService_name(self, ctx:apgParser.Service_nameContext):
        pass

    # Exit a parse tree produced by apgParser#service_name.
    def exitService_name(self, ctx:apgParser.Service_nameContext):
        pass


    # Enter a parse tree produced by apgParser#service_type.
    def enterService_type(self, ctx:apgParser.Service_typeContext):
        pass

    # Exit a parse tree produced by apgParser#service_type.
    def exitService_type(self, ctx:apgParser.Service_typeContext):
        pass


    # Enter a parse tree produced by apgParser#shipping_zones_config.
    def enterShipping_zones_config(self, ctx:apgParser.Shipping_zones_configContext):
        pass

    # Exit a parse tree produced by apgParser#shipping_zones_config.
    def exitShipping_zones_config(self, ctx:apgParser.Shipping_zones_configContext):
        pass


    # Enter a parse tree produced by apgParser#split_configuration.
    def enterSplit_configuration(self, ctx:apgParser.Split_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#split_configuration.
    def exitSplit_configuration(self, ctx:apgParser.Split_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#storage_configuration.
    def enterStorage_configuration(self, ctx:apgParser.Storage_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#storage_configuration.
    def exitStorage_configuration(self, ctx:apgParser.Storage_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#suppression_rules.
    def enterSuppression_rules(self, ctx:apgParser.Suppression_rulesContext):
        pass

    # Exit a parse tree produced by apgParser#suppression_rules.
    def exitSuppression_rules(self, ctx:apgParser.Suppression_rulesContext):
        pass


    # Enter a parse tree produced by apgParser#threshold_value.
    def enterThreshold_value(self, ctx:apgParser.Threshold_valueContext):
        pass

    # Exit a parse tree produced by apgParser#threshold_value.
    def exitThreshold_value(self, ctx:apgParser.Threshold_valueContext):
        pass


    # Enter a parse tree produced by apgParser#tracing_config.
    def enterTracing_config(self, ctx:apgParser.Tracing_configContext):
        pass

    # Exit a parse tree produced by apgParser#tracing_config.
    def exitTracing_config(self, ctx:apgParser.Tracing_configContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_list.
    def enterTrigger_list(self, ctx:apgParser.Trigger_listContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_list.
    def exitTrigger_list(self, ctx:apgParser.Trigger_listContext):
        pass


    # Enter a parse tree produced by apgParser#user_type_name.
    def enterUser_type_name(self, ctx:apgParser.User_type_nameContext):
        pass

    # Exit a parse tree produced by apgParser#user_type_name.
    def exitUser_type_name(self, ctx:apgParser.User_type_nameContext):
        pass


    # Enter a parse tree produced by apgParser#verification_config.
    def enterVerification_config(self, ctx:apgParser.Verification_configContext):
        pass

    # Exit a parse tree produced by apgParser#verification_config.
    def exitVerification_config(self, ctx:apgParser.Verification_configContext):
        pass


    # Enter a parse tree produced by apgParser#verification_requirements.
    def enterVerification_requirements(self, ctx:apgParser.Verification_requirementsContext):
        pass

    # Exit a parse tree produced by apgParser#verification_requirements.
    def exitVerification_requirements(self, ctx:apgParser.Verification_requirementsContext):
        pass


    # Enter a parse tree produced by apgParser#video_call_configuration.
    def enterVideo_call_configuration(self, ctx:apgParser.Video_call_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#video_call_configuration.
    def exitVideo_call_configuration(self, ctx:apgParser.Video_call_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#form_layout.
    def enterForm_layout(self, ctx:apgParser.Form_layoutContext):
        pass

    # Exit a parse tree produced by apgParser#form_layout.
    def exitForm_layout(self, ctx:apgParser.Form_layoutContext):
        pass


    # Enter a parse tree produced by apgParser#layout_type.
    def enterLayout_type(self, ctx:apgParser.Layout_typeContext):
        pass

    # Exit a parse tree produced by apgParser#layout_type.
    def exitLayout_type(self, ctx:apgParser.Layout_typeContext):
        pass


    # Enter a parse tree produced by apgParser#layout_definition.
    def enterLayout_definition(self, ctx:apgParser.Layout_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#layout_definition.
    def exitLayout_definition(self, ctx:apgParser.Layout_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#layout_element.
    def enterLayout_element(self, ctx:apgParser.Layout_elementContext):
        pass

    # Exit a parse tree produced by apgParser#layout_element.
    def exitLayout_element(self, ctx:apgParser.Layout_elementContext):
        pass


    # Enter a parse tree produced by apgParser#container_element.
    def enterContainer_element(self, ctx:apgParser.Container_elementContext):
        pass

    # Exit a parse tree produced by apgParser#container_element.
    def exitContainer_element(self, ctx:apgParser.Container_elementContext):
        pass


    # Enter a parse tree produced by apgParser#container_type.
    def enterContainer_type(self, ctx:apgParser.Container_typeContext):
        pass

    # Exit a parse tree produced by apgParser#container_type.
    def exitContainer_type(self, ctx:apgParser.Container_typeContext):
        pass


    # Enter a parse tree produced by apgParser#field_element.
    def enterField_element(self, ctx:apgParser.Field_elementContext):
        pass

    # Exit a parse tree produced by apgParser#field_element.
    def exitField_element(self, ctx:apgParser.Field_elementContext):
        pass


    # Enter a parse tree produced by apgParser#field_type.
    def enterField_type(self, ctx:apgParser.Field_typeContext):
        pass

    # Exit a parse tree produced by apgParser#field_type.
    def exitField_type(self, ctx:apgParser.Field_typeContext):
        pass


    # Enter a parse tree produced by apgParser#component_element.
    def enterComponent_element(self, ctx:apgParser.Component_elementContext):
        pass

    # Exit a parse tree produced by apgParser#component_element.
    def exitComponent_element(self, ctx:apgParser.Component_elementContext):
        pass


    # Enter a parse tree produced by apgParser#component_type.
    def enterComponent_type(self, ctx:apgParser.Component_typeContext):
        pass

    # Exit a parse tree produced by apgParser#component_type.
    def exitComponent_type(self, ctx:apgParser.Component_typeContext):
        pass


    # Enter a parse tree produced by apgParser#layout_directive.
    def enterLayout_directive(self, ctx:apgParser.Layout_directiveContext):
        pass

    # Exit a parse tree produced by apgParser#layout_directive.
    def exitLayout_directive(self, ctx:apgParser.Layout_directiveContext):
        pass


    # Enter a parse tree produced by apgParser#directive_name.
    def enterDirective_name(self, ctx:apgParser.Directive_nameContext):
        pass

    # Exit a parse tree produced by apgParser#directive_name.
    def exitDirective_name(self, ctx:apgParser.Directive_nameContext):
        pass


    # Enter a parse tree produced by apgParser#directive_params.
    def enterDirective_params(self, ctx:apgParser.Directive_paramsContext):
        pass

    # Exit a parse tree produced by apgParser#directive_params.
    def exitDirective_params(self, ctx:apgParser.Directive_paramsContext):
        pass


    # Enter a parse tree produced by apgParser#directive_param.
    def enterDirective_param(self, ctx:apgParser.Directive_paramContext):
        pass

    # Exit a parse tree produced by apgParser#directive_param.
    def exitDirective_param(self, ctx:apgParser.Directive_paramContext):
        pass


    # Enter a parse tree produced by apgParser#layout_properties.
    def enterLayout_properties(self, ctx:apgParser.Layout_propertiesContext):
        pass

    # Exit a parse tree produced by apgParser#layout_properties.
    def exitLayout_properties(self, ctx:apgParser.Layout_propertiesContext):
        pass


    # Enter a parse tree produced by apgParser#layout_property.
    def enterLayout_property(self, ctx:apgParser.Layout_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#layout_property.
    def exitLayout_property(self, ctx:apgParser.Layout_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#style_property.
    def enterStyle_property(self, ctx:apgParser.Style_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#style_property.
    def exitStyle_property(self, ctx:apgParser.Style_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#field_properties.
    def enterField_properties(self, ctx:apgParser.Field_propertiesContext):
        pass

    # Exit a parse tree produced by apgParser#field_properties.
    def exitField_properties(self, ctx:apgParser.Field_propertiesContext):
        pass


    # Enter a parse tree produced by apgParser#field_property.
    def enterField_property(self, ctx:apgParser.Field_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#field_property.
    def exitField_property(self, ctx:apgParser.Field_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#component_properties.
    def enterComponent_properties(self, ctx:apgParser.Component_propertiesContext):
        pass

    # Exit a parse tree produced by apgParser#component_properties.
    def exitComponent_properties(self, ctx:apgParser.Component_propertiesContext):
        pass


    # Enter a parse tree produced by apgParser#component_property.
    def enterComponent_property(self, ctx:apgParser.Component_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#component_property.
    def exitComponent_property(self, ctx:apgParser.Component_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#size_value.
    def enterSize_value(self, ctx:apgParser.Size_valueContext):
        pass

    # Exit a parse tree produced by apgParser#size_value.
    def exitSize_value(self, ctx:apgParser.Size_valueContext):
        pass


    # Enter a parse tree produced by apgParser#spacing_value.
    def enterSpacing_value(self, ctx:apgParser.Spacing_valueContext):
        pass

    # Exit a parse tree produced by apgParser#spacing_value.
    def exitSpacing_value(self, ctx:apgParser.Spacing_valueContext):
        pass


    # Enter a parse tree produced by apgParser#color_value.
    def enterColor_value(self, ctx:apgParser.Color_valueContext):
        pass

    # Exit a parse tree produced by apgParser#color_value.
    def exitColor_value(self, ctx:apgParser.Color_valueContext):
        pass


    # Enter a parse tree produced by apgParser#border_value.
    def enterBorder_value(self, ctx:apgParser.Border_valueContext):
        pass

    # Exit a parse tree produced by apgParser#border_value.
    def exitBorder_value(self, ctx:apgParser.Border_valueContext):
        pass


    # Enter a parse tree produced by apgParser#font_value.
    def enterFont_value(self, ctx:apgParser.Font_valueContext):
        pass

    # Exit a parse tree produced by apgParser#font_value.
    def exitFont_value(self, ctx:apgParser.Font_valueContext):
        pass


    # Enter a parse tree produced by apgParser#display_value.
    def enterDisplay_value(self, ctx:apgParser.Display_valueContext):
        pass

    # Exit a parse tree produced by apgParser#display_value.
    def exitDisplay_value(self, ctx:apgParser.Display_valueContext):
        pass


    # Enter a parse tree produced by apgParser#position_value.
    def enterPosition_value(self, ctx:apgParser.Position_valueContext):
        pass

    # Exit a parse tree produced by apgParser#position_value.
    def exitPosition_value(self, ctx:apgParser.Position_valueContext):
        pass


    # Enter a parse tree produced by apgParser#animation_value.
    def enterAnimation_value(self, ctx:apgParser.Animation_valueContext):
        pass

    # Exit a parse tree produced by apgParser#animation_value.
    def exitAnimation_value(self, ctx:apgParser.Animation_valueContext):
        pass


    # Enter a parse tree produced by apgParser#transition_value.
    def enterTransition_value(self, ctx:apgParser.Transition_valueContext):
        pass

    # Exit a parse tree produced by apgParser#transition_value.
    def exitTransition_value(self, ctx:apgParser.Transition_valueContext):
        pass


    # Enter a parse tree produced by apgParser#responsive_breakpoints.
    def enterResponsive_breakpoints(self, ctx:apgParser.Responsive_breakpointsContext):
        pass

    # Exit a parse tree produced by apgParser#responsive_breakpoints.
    def exitResponsive_breakpoints(self, ctx:apgParser.Responsive_breakpointsContext):
        pass


    # Enter a parse tree produced by apgParser#media_query.
    def enterMedia_query(self, ctx:apgParser.Media_queryContext):
        pass

    # Exit a parse tree produced by apgParser#media_query.
    def exitMedia_query(self, ctx:apgParser.Media_queryContext):
        pass


    # Enter a parse tree produced by apgParser#media_type.
    def enterMedia_type(self, ctx:apgParser.Media_typeContext):
        pass

    # Exit a parse tree produced by apgParser#media_type.
    def exitMedia_type(self, ctx:apgParser.Media_typeContext):
        pass


    # Enter a parse tree produced by apgParser#media_features.
    def enterMedia_features(self, ctx:apgParser.Media_featuresContext):
        pass

    # Exit a parse tree produced by apgParser#media_features.
    def exitMedia_features(self, ctx:apgParser.Media_featuresContext):
        pass


    # Enter a parse tree produced by apgParser#media_feature.
    def enterMedia_feature(self, ctx:apgParser.Media_featureContext):
        pass

    # Exit a parse tree produced by apgParser#media_feature.
    def exitMedia_feature(self, ctx:apgParser.Media_featureContext):
        pass


    # Enter a parse tree produced by apgParser#validation_rules.
    def enterValidation_rules(self, ctx:apgParser.Validation_rulesContext):
        pass

    # Exit a parse tree produced by apgParser#validation_rules.
    def exitValidation_rules(self, ctx:apgParser.Validation_rulesContext):
        pass


    # Enter a parse tree produced by apgParser#validation_rule.
    def enterValidation_rule(self, ctx:apgParser.Validation_ruleContext):
        pass

    # Exit a parse tree produced by apgParser#validation_rule.
    def exitValidation_rule(self, ctx:apgParser.Validation_ruleContext):
        pass


    # Enter a parse tree produced by apgParser#cross_field_validation.
    def enterCross_field_validation(self, ctx:apgParser.Cross_field_validationContext):
        pass

    # Exit a parse tree produced by apgParser#cross_field_validation.
    def exitCross_field_validation(self, ctx:apgParser.Cross_field_validationContext):
        pass


    # Enter a parse tree produced by apgParser#field_comparison.
    def enterField_comparison(self, ctx:apgParser.Field_comparisonContext):
        pass

    # Exit a parse tree produced by apgParser#field_comparison.
    def exitField_comparison(self, ctx:apgParser.Field_comparisonContext):
        pass


    # Enter a parse tree produced by apgParser#comparison_operator.
    def enterComparison_operator(self, ctx:apgParser.Comparison_operatorContext):
        pass

    # Exit a parse tree produced by apgParser#comparison_operator.
    def exitComparison_operator(self, ctx:apgParser.Comparison_operatorContext):
        pass


    # Enter a parse tree produced by apgParser#behavior_property.
    def enterBehavior_property(self, ctx:apgParser.Behavior_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#behavior_property.
    def exitBehavior_property(self, ctx:apgParser.Behavior_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#data_property.
    def enterData_property(self, ctx:apgParser.Data_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#data_property.
    def exitData_property(self, ctx:apgParser.Data_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#accessibility_property.
    def enterAccessibility_property(self, ctx:apgParser.Accessibility_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#accessibility_property.
    def exitAccessibility_property(self, ctx:apgParser.Accessibility_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#transform_value.
    def enterTransform_value(self, ctx:apgParser.Transform_valueContext):
        pass

    # Exit a parse tree produced by apgParser#transform_value.
    def exitTransform_value(self, ctx:apgParser.Transform_valueContext):
        pass


    # Enter a parse tree produced by apgParser#transform_function.
    def enterTransform_function(self, ctx:apgParser.Transform_functionContext):
        pass

    # Exit a parse tree produced by apgParser#transform_function.
    def exitTransform_function(self, ctx:apgParser.Transform_functionContext):
        pass


    # Enter a parse tree produced by apgParser#filter_value.
    def enterFilter_value(self, ctx:apgParser.Filter_valueContext):
        pass

    # Exit a parse tree produced by apgParser#filter_value.
    def exitFilter_value(self, ctx:apgParser.Filter_valueContext):
        pass


    # Enter a parse tree produced by apgParser#filter_function.
    def enterFilter_function(self, ctx:apgParser.Filter_functionContext):
        pass

    # Exit a parse tree produced by apgParser#filter_function.
    def exitFilter_function(self, ctx:apgParser.Filter_functionContext):
        pass


    # Enter a parse tree produced by apgParser#shadow_value.
    def enterShadow_value(self, ctx:apgParser.Shadow_valueContext):
        pass

    # Exit a parse tree produced by apgParser#shadow_value.
    def exitShadow_value(self, ctx:apgParser.Shadow_valueContext):
        pass


    # Enter a parse tree produced by apgParser#gradient_value.
    def enterGradient_value(self, ctx:apgParser.Gradient_valueContext):
        pass

    # Exit a parse tree produced by apgParser#gradient_value.
    def exitGradient_value(self, ctx:apgParser.Gradient_valueContext):
        pass


    # Enter a parse tree produced by apgParser#gradient_direction.
    def enterGradient_direction(self, ctx:apgParser.Gradient_directionContext):
        pass

    # Exit a parse tree produced by apgParser#gradient_direction.
    def exitGradient_direction(self, ctx:apgParser.Gradient_directionContext):
        pass


    # Enter a parse tree produced by apgParser#gradient_stops.
    def enterGradient_stops(self, ctx:apgParser.Gradient_stopsContext):
        pass

    # Exit a parse tree produced by apgParser#gradient_stops.
    def exitGradient_stops(self, ctx:apgParser.Gradient_stopsContext):
        pass


    # Enter a parse tree produced by apgParser#gradient_stop.
    def enterGradient_stop(self, ctx:apgParser.Gradient_stopContext):
        pass

    # Exit a parse tree produced by apgParser#gradient_stop.
    def exitGradient_stop(self, ctx:apgParser.Gradient_stopContext):
        pass


    # Enter a parse tree produced by apgParser#clip_value.
    def enterClip_value(self, ctx:apgParser.Clip_valueContext):
        pass

    # Exit a parse tree produced by apgParser#clip_value.
    def exitClip_value(self, ctx:apgParser.Clip_valueContext):
        pass


    # Enter a parse tree produced by apgParser#clip_points.
    def enterClip_points(self, ctx:apgParser.Clip_pointsContext):
        pass

    # Exit a parse tree produced by apgParser#clip_points.
    def exitClip_points(self, ctx:apgParser.Clip_pointsContext):
        pass


    # Enter a parse tree produced by apgParser#clip_point.
    def enterClip_point(self, ctx:apgParser.Clip_pointContext):
        pass

    # Exit a parse tree produced by apgParser#clip_point.
    def exitClip_point(self, ctx:apgParser.Clip_pointContext):
        pass


    # Enter a parse tree produced by apgParser#mask_value.
    def enterMask_value(self, ctx:apgParser.Mask_valueContext):
        pass

    # Exit a parse tree produced by apgParser#mask_value.
    def exitMask_value(self, ctx:apgParser.Mask_valueContext):
        pass


    # Enter a parse tree produced by apgParser#format_specification.
    def enterFormat_specification(self, ctx:apgParser.Format_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#format_specification.
    def exitFormat_specification(self, ctx:apgParser.Format_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#input_mask.
    def enterInput_mask(self, ctx:apgParser.Input_maskContext):
        pass

    # Exit a parse tree produced by apgParser#input_mask.
    def exitInput_mask(self, ctx:apgParser.Input_maskContext):
        pass


    # Enter a parse tree produced by apgParser#mask_pattern.
    def enterMask_pattern(self, ctx:apgParser.Mask_patternContext):
        pass

    # Exit a parse tree produced by apgParser#mask_pattern.
    def exitMask_pattern(self, ctx:apgParser.Mask_patternContext):
        pass


    # Enter a parse tree produced by apgParser#mask_options.
    def enterMask_options(self, ctx:apgParser.Mask_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#mask_options.
    def exitMask_options(self, ctx:apgParser.Mask_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#autocomplete_spec.
    def enterAutocomplete_spec(self, ctx:apgParser.Autocomplete_specContext):
        pass

    # Exit a parse tree produced by apgParser#autocomplete_spec.
    def exitAutocomplete_spec(self, ctx:apgParser.Autocomplete_specContext):
        pass


    # Enter a parse tree produced by apgParser#autocomplete_source.
    def enterAutocomplete_source(self, ctx:apgParser.Autocomplete_sourceContext):
        pass

    # Exit a parse tree produced by apgParser#autocomplete_source.
    def exitAutocomplete_source(self, ctx:apgParser.Autocomplete_sourceContext):
        pass


    # Enter a parse tree produced by apgParser#autocomplete_options.
    def enterAutocomplete_options(self, ctx:apgParser.Autocomplete_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#autocomplete_options.
    def exitAutocomplete_options(self, ctx:apgParser.Autocomplete_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#dependency_spec.
    def enterDependency_spec(self, ctx:apgParser.Dependency_specContext):
        pass

    # Exit a parse tree produced by apgParser#dependency_spec.
    def exitDependency_spec(self, ctx:apgParser.Dependency_specContext):
        pass


    # Enter a parse tree produced by apgParser#conditional_spec.
    def enterConditional_spec(self, ctx:apgParser.Conditional_specContext):
        pass

    # Exit a parse tree produced by apgParser#conditional_spec.
    def exitConditional_spec(self, ctx:apgParser.Conditional_specContext):
        pass


    # Enter a parse tree produced by apgParser#accessibility_spec.
    def enterAccessibility_spec(self, ctx:apgParser.Accessibility_specContext):
        pass

    # Exit a parse tree produced by apgParser#accessibility_spec.
    def exitAccessibility_spec(self, ctx:apgParser.Accessibility_specContext):
        pass


    # Enter a parse tree produced by apgParser#accessibility_rule.
    def enterAccessibility_rule(self, ctx:apgParser.Accessibility_ruleContext):
        pass

    # Exit a parse tree produced by apgParser#accessibility_rule.
    def exitAccessibility_rule(self, ctx:apgParser.Accessibility_ruleContext):
        pass


    # Enter a parse tree produced by apgParser#angle_value.
    def enterAngle_value(self, ctx:apgParser.Angle_valueContext):
        pass

    # Exit a parse tree produced by apgParser#angle_value.
    def exitAngle_value(self, ctx:apgParser.Angle_valueContext):
        pass


    # Enter a parse tree produced by apgParser#percentage_value.
    def enterPercentage_value(self, ctx:apgParser.Percentage_valueContext):
        pass

    # Exit a parse tree produced by apgParser#percentage_value.
    def exitPercentage_value(self, ctx:apgParser.Percentage_valueContext):
        pass


    # Enter a parse tree produced by apgParser#viewport_value.
    def enterViewport_value(self, ctx:apgParser.Viewport_valueContext):
        pass

    # Exit a parse tree produced by apgParser#viewport_value.
    def exitViewport_value(self, ctx:apgParser.Viewport_valueContext):
        pass


    # Enter a parse tree produced by apgParser#calc_expression.
    def enterCalc_expression(self, ctx:apgParser.Calc_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#calc_expression.
    def exitCalc_expression(self, ctx:apgParser.Calc_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#calc_operand.
    def enterCalc_operand(self, ctx:apgParser.Calc_operandContext):
        pass

    # Exit a parse tree produced by apgParser#calc_operand.
    def exitCalc_operand(self, ctx:apgParser.Calc_operandContext):
        pass


    # Enter a parse tree produced by apgParser#calc_operator.
    def enterCalc_operator(self, ctx:apgParser.Calc_operatorContext):
        pass

    # Exit a parse tree produced by apgParser#calc_operator.
    def exitCalc_operator(self, ctx:apgParser.Calc_operatorContext):
        pass


    # Enter a parse tree produced by apgParser#icon_specification.
    def enterIcon_specification(self, ctx:apgParser.Icon_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#icon_specification.
    def exitIcon_specification(self, ctx:apgParser.Icon_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#icon_name.
    def enterIcon_name(self, ctx:apgParser.Icon_nameContext):
        pass

    # Exit a parse tree produced by apgParser#icon_name.
    def exitIcon_name(self, ctx:apgParser.Icon_nameContext):
        pass


    # Enter a parse tree produced by apgParser#icon_style.
    def enterIcon_style(self, ctx:apgParser.Icon_styleContext):
        pass

    # Exit a parse tree produced by apgParser#icon_style.
    def exitIcon_style(self, ctx:apgParser.Icon_styleContext):
        pass


    # Enter a parse tree produced by apgParser#icon_size.
    def enterIcon_size(self, ctx:apgParser.Icon_sizeContext):
        pass

    # Exit a parse tree produced by apgParser#icon_size.
    def exitIcon_size(self, ctx:apgParser.Icon_sizeContext):
        pass


    # Enter a parse tree produced by apgParser#action_specification.
    def enterAction_specification(self, ctx:apgParser.Action_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#action_specification.
    def exitAction_specification(self, ctx:apgParser.Action_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#function_call.
    def enterFunction_call(self, ctx:apgParser.Function_callContext):
        pass

    # Exit a parse tree produced by apgParser#function_call.
    def exitFunction_call(self, ctx:apgParser.Function_callContext):
        pass


    # Enter a parse tree produced by apgParser#event_handler.
    def enterEvent_handler(self, ctx:apgParser.Event_handlerContext):
        pass

    # Exit a parse tree produced by apgParser#event_handler.
    def exitEvent_handler(self, ctx:apgParser.Event_handlerContext):
        pass


    # Enter a parse tree produced by apgParser#route_action.
    def enterRoute_action(self, ctx:apgParser.Route_actionContext):
        pass

    # Exit a parse tree produced by apgParser#route_action.
    def exitRoute_action(self, ctx:apgParser.Route_actionContext):
        pass


    # Enter a parse tree produced by apgParser#custom_action.
    def enterCustom_action(self, ctx:apgParser.Custom_actionContext):
        pass

    # Exit a parse tree produced by apgParser#custom_action.
    def exitCustom_action(self, ctx:apgParser.Custom_actionContext):
        pass


    # Enter a parse tree produced by apgParser#action_property.
    def enterAction_property(self, ctx:apgParser.Action_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#action_property.
    def exitAction_property(self, ctx:apgParser.Action_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#data_specification.
    def enterData_specification(self, ctx:apgParser.Data_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#data_specification.
    def exitData_specification(self, ctx:apgParser.Data_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#data_binding.
    def enterData_binding(self, ctx:apgParser.Data_bindingContext):
        pass

    # Exit a parse tree produced by apgParser#data_binding.
    def exitData_binding(self, ctx:apgParser.Data_bindingContext):
        pass


    # Enter a parse tree produced by apgParser#data_source.
    def enterData_source(self, ctx:apgParser.Data_sourceContext):
        pass

    # Exit a parse tree produced by apgParser#data_source.
    def exitData_source(self, ctx:apgParser.Data_sourceContext):
        pass


    # Enter a parse tree produced by apgParser#data_model.
    def enterData_model(self, ctx:apgParser.Data_modelContext):
        pass

    # Exit a parse tree produced by apgParser#data_model.
    def exitData_model(self, ctx:apgParser.Data_modelContext):
        pass


    # Enter a parse tree produced by apgParser#state_specification.
    def enterState_specification(self, ctx:apgParser.State_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#state_specification.
    def exitState_specification(self, ctx:apgParser.State_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#state_property.
    def enterState_property(self, ctx:apgParser.State_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#state_property.
    def exitState_property(self, ctx:apgParser.State_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#theme_specification.
    def enterTheme_specification(self, ctx:apgParser.Theme_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#theme_specification.
    def exitTheme_specification(self, ctx:apgParser.Theme_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#theme_definition.
    def enterTheme_definition(self, ctx:apgParser.Theme_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#theme_definition.
    def exitTheme_definition(self, ctx:apgParser.Theme_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#theme_property.
    def enterTheme_property(self, ctx:apgParser.Theme_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#theme_property.
    def exitTheme_property(self, ctx:apgParser.Theme_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#variant_specification.
    def enterVariant_specification(self, ctx:apgParser.Variant_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#variant_specification.
    def exitVariant_specification(self, ctx:apgParser.Variant_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#size_specification.
    def enterSize_specification(self, ctx:apgParser.Size_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#size_specification.
    def exitSize_specification(self, ctx:apgParser.Size_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#alignment_specification.
    def enterAlignment_specification(self, ctx:apgParser.Alignment_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#alignment_specification.
    def exitAlignment_specification(self, ctx:apgParser.Alignment_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#unit.
    def enterUnit(self, ctx:apgParser.UnitContext):
        pass

    # Exit a parse tree produced by apgParser#unit.
    def exitUnit(self, ctx:apgParser.UnitContext):
        pass


    # Enter a parse tree produced by apgParser#hex_color.
    def enterHex_color(self, ctx:apgParser.Hex_colorContext):
        pass

    # Exit a parse tree produced by apgParser#hex_color.
    def exitHex_color(self, ctx:apgParser.Hex_colorContext):
        pass


    # Enter a parse tree produced by apgParser#rgb_color.
    def enterRgb_color(self, ctx:apgParser.Rgb_colorContext):
        pass

    # Exit a parse tree produced by apgParser#rgb_color.
    def exitRgb_color(self, ctx:apgParser.Rgb_colorContext):
        pass


    # Enter a parse tree produced by apgParser#hsl_color.
    def enterHsl_color(self, ctx:apgParser.Hsl_colorContext):
        pass

    # Exit a parse tree produced by apgParser#hsl_color.
    def exitHsl_color(self, ctx:apgParser.Hsl_colorContext):
        pass


    # Enter a parse tree produced by apgParser#named_color.
    def enterNamed_color(self, ctx:apgParser.Named_colorContext):
        pass

    # Exit a parse tree produced by apgParser#named_color.
    def exitNamed_color(self, ctx:apgParser.Named_colorContext):
        pass


    # Enter a parse tree produced by apgParser#css_variable.
    def enterCss_variable(self, ctx:apgParser.Css_variableContext):
        pass

    # Exit a parse tree produced by apgParser#css_variable.
    def exitCss_variable(self, ctx:apgParser.Css_variableContext):
        pass


    # Enter a parse tree produced by apgParser#border_width.
    def enterBorder_width(self, ctx:apgParser.Border_widthContext):
        pass

    # Exit a parse tree produced by apgParser#border_width.
    def exitBorder_width(self, ctx:apgParser.Border_widthContext):
        pass


    # Enter a parse tree produced by apgParser#border_style.
    def enterBorder_style(self, ctx:apgParser.Border_styleContext):
        pass

    # Exit a parse tree produced by apgParser#border_style.
    def exitBorder_style(self, ctx:apgParser.Border_styleContext):
        pass


    # Enter a parse tree produced by apgParser#border_color.
    def enterBorder_color(self, ctx:apgParser.Border_colorContext):
        pass

    # Exit a parse tree produced by apgParser#border_color.
    def exitBorder_color(self, ctx:apgParser.Border_colorContext):
        pass


    # Enter a parse tree produced by apgParser#font_family.
    def enterFont_family(self, ctx:apgParser.Font_familyContext):
        pass

    # Exit a parse tree produced by apgParser#font_family.
    def exitFont_family(self, ctx:apgParser.Font_familyContext):
        pass


    # Enter a parse tree produced by apgParser#font_family_name.
    def enterFont_family_name(self, ctx:apgParser.Font_family_nameContext):
        pass

    # Exit a parse tree produced by apgParser#font_family_name.
    def exitFont_family_name(self, ctx:apgParser.Font_family_nameContext):
        pass


    # Enter a parse tree produced by apgParser#font_size.
    def enterFont_size(self, ctx:apgParser.Font_sizeContext):
        pass

    # Exit a parse tree produced by apgParser#font_size.
    def exitFont_size(self, ctx:apgParser.Font_sizeContext):
        pass


    # Enter a parse tree produced by apgParser#font_size_keyword.
    def enterFont_size_keyword(self, ctx:apgParser.Font_size_keywordContext):
        pass

    # Exit a parse tree produced by apgParser#font_size_keyword.
    def exitFont_size_keyword(self, ctx:apgParser.Font_size_keywordContext):
        pass


    # Enter a parse tree produced by apgParser#font_weight.
    def enterFont_weight(self, ctx:apgParser.Font_weightContext):
        pass

    # Exit a parse tree produced by apgParser#font_weight.
    def exitFont_weight(self, ctx:apgParser.Font_weightContext):
        pass


    # Enter a parse tree produced by apgParser#font_weight_keyword.
    def enterFont_weight_keyword(self, ctx:apgParser.Font_weight_keywordContext):
        pass

    # Exit a parse tree produced by apgParser#font_weight_keyword.
    def exitFont_weight_keyword(self, ctx:apgParser.Font_weight_keywordContext):
        pass


    # Enter a parse tree produced by apgParser#font_style.
    def enterFont_style(self, ctx:apgParser.Font_styleContext):
        pass

    # Exit a parse tree produced by apgParser#font_style.
    def exitFont_style(self, ctx:apgParser.Font_styleContext):
        pass


    # Enter a parse tree produced by apgParser#animation_name.
    def enterAnimation_name(self, ctx:apgParser.Animation_nameContext):
        pass

    # Exit a parse tree produced by apgParser#animation_name.
    def exitAnimation_name(self, ctx:apgParser.Animation_nameContext):
        pass


    # Enter a parse tree produced by apgParser#duration.
    def enterDuration(self, ctx:apgParser.DurationContext):
        pass

    # Exit a parse tree produced by apgParser#duration.
    def exitDuration(self, ctx:apgParser.DurationContext):
        pass


    # Enter a parse tree produced by apgParser#timing_function.
    def enterTiming_function(self, ctx:apgParser.Timing_functionContext):
        pass

    # Exit a parse tree produced by apgParser#timing_function.
    def exitTiming_function(self, ctx:apgParser.Timing_functionContext):
        pass


    # Enter a parse tree produced by apgParser#delay.
    def enterDelay(self, ctx:apgParser.DelayContext):
        pass

    # Exit a parse tree produced by apgParser#delay.
    def exitDelay(self, ctx:apgParser.DelayContext):
        pass


    # Enter a parse tree produced by apgParser#iteration_count.
    def enterIteration_count(self, ctx:apgParser.Iteration_countContext):
        pass

    # Exit a parse tree produced by apgParser#iteration_count.
    def exitIteration_count(self, ctx:apgParser.Iteration_countContext):
        pass


    # Enter a parse tree produced by apgParser#direction.
    def enterDirection(self, ctx:apgParser.DirectionContext):
        pass

    # Exit a parse tree produced by apgParser#direction.
    def exitDirection(self, ctx:apgParser.DirectionContext):
        pass


    # Enter a parse tree produced by apgParser#fill_mode.
    def enterFill_mode(self, ctx:apgParser.Fill_modeContext):
        pass

    # Exit a parse tree produced by apgParser#fill_mode.
    def exitFill_mode(self, ctx:apgParser.Fill_modeContext):
        pass


    # Enter a parse tree produced by apgParser#transition_property.
    def enterTransition_property(self, ctx:apgParser.Transition_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#transition_property.
    def exitTransition_property(self, ctx:apgParser.Transition_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#resolution_value.
    def enterResolution_value(self, ctx:apgParser.Resolution_valueContext):
        pass

    # Exit a parse tree produced by apgParser#resolution_value.
    def exitResolution_value(self, ctx:apgParser.Resolution_valueContext):
        pass


    # Enter a parse tree produced by apgParser#custom_validation_function.
    def enterCustom_validation_function(self, ctx:apgParser.Custom_validation_functionContext):
        pass

    # Exit a parse tree produced by apgParser#custom_validation_function.
    def exitCustom_validation_function(self, ctx:apgParser.Custom_validation_functionContext):
        pass


    # Enter a parse tree produced by apgParser#data_binding_spec.
    def enterData_binding_spec(self, ctx:apgParser.Data_binding_specContext):
        pass

    # Exit a parse tree produced by apgParser#data_binding_spec.
    def exitData_binding_spec(self, ctx:apgParser.Data_binding_specContext):
        pass


    # Enter a parse tree produced by apgParser#simple_binding.
    def enterSimple_binding(self, ctx:apgParser.Simple_bindingContext):
        pass

    # Exit a parse tree produced by apgParser#simple_binding.
    def exitSimple_binding(self, ctx:apgParser.Simple_bindingContext):
        pass


    # Enter a parse tree produced by apgParser#complex_binding.
    def enterComplex_binding(self, ctx:apgParser.Complex_bindingContext):
        pass

    # Exit a parse tree produced by apgParser#complex_binding.
    def exitComplex_binding(self, ctx:apgParser.Complex_bindingContext):
        pass


    # Enter a parse tree produced by apgParser#binding_property.
    def enterBinding_property(self, ctx:apgParser.Binding_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#binding_property.
    def exitBinding_property(self, ctx:apgParser.Binding_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#data_source_spec.
    def enterData_source_spec(self, ctx:apgParser.Data_source_specContext):
        pass

    # Exit a parse tree produced by apgParser#data_source_spec.
    def exitData_source_spec(self, ctx:apgParser.Data_source_specContext):
        pass


    # Enter a parse tree produced by apgParser#static_data.
    def enterStatic_data(self, ctx:apgParser.Static_dataContext):
        pass

    # Exit a parse tree produced by apgParser#static_data.
    def exitStatic_data(self, ctx:apgParser.Static_dataContext):
        pass


    # Enter a parse tree produced by apgParser#computed_data.
    def enterComputed_data(self, ctx:apgParser.Computed_dataContext):
        pass

    # Exit a parse tree produced by apgParser#computed_data.
    def exitComputed_data(self, ctx:apgParser.Computed_dataContext):
        pass


    # Enter a parse tree produced by apgParser#computed_expression.
    def enterComputed_expression(self, ctx:apgParser.Computed_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#computed_expression.
    def exitComputed_expression(self, ctx:apgParser.Computed_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#lambda_expression.
    def enterLambda_expression(self, ctx:apgParser.Lambda_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#lambda_expression.
    def exitLambda_expression(self, ctx:apgParser.Lambda_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#watcher_spec.
    def enterWatcher_spec(self, ctx:apgParser.Watcher_specContext):
        pass

    # Exit a parse tree produced by apgParser#watcher_spec.
    def exitWatcher_spec(self, ctx:apgParser.Watcher_specContext):
        pass


    # Enter a parse tree produced by apgParser#watch_property.
    def enterWatch_property(self, ctx:apgParser.Watch_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#watch_property.
    def exitWatch_property(self, ctx:apgParser.Watch_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#cache_spec.
    def enterCache_spec(self, ctx:apgParser.Cache_specContext):
        pass

    # Exit a parse tree produced by apgParser#cache_spec.
    def exitCache_spec(self, ctx:apgParser.Cache_specContext):
        pass


    # Enter a parse tree produced by apgParser#cache_property.
    def enterCache_property(self, ctx:apgParser.Cache_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#cache_property.
    def exitCache_property(self, ctx:apgParser.Cache_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#gradient_shape.
    def enterGradient_shape(self, ctx:apgParser.Gradient_shapeContext):
        pass

    # Exit a parse tree produced by apgParser#gradient_shape.
    def exitGradient_shape(self, ctx:apgParser.Gradient_shapeContext):
        pass


    # Enter a parse tree produced by apgParser#gradient_angle.
    def enterGradient_angle(self, ctx:apgParser.Gradient_angleContext):
        pass

    # Exit a parse tree produced by apgParser#gradient_angle.
    def exitGradient_angle(self, ctx:apgParser.Gradient_angleContext):
        pass


    # Enter a parse tree produced by apgParser#mask_source.
    def enterMask_source(self, ctx:apgParser.Mask_sourceContext):
        pass

    # Exit a parse tree produced by apgParser#mask_source.
    def exitMask_source(self, ctx:apgParser.Mask_sourceContext):
        pass


    # Enter a parse tree produced by apgParser#mask_position.
    def enterMask_position(self, ctx:apgParser.Mask_positionContext):
        pass

    # Exit a parse tree produced by apgParser#mask_position.
    def exitMask_position(self, ctx:apgParser.Mask_positionContext):
        pass


    # Enter a parse tree produced by apgParser#mask_size.
    def enterMask_size(self, ctx:apgParser.Mask_sizeContext):
        pass

    # Exit a parse tree produced by apgParser#mask_size.
    def exitMask_size(self, ctx:apgParser.Mask_sizeContext):
        pass


    # Enter a parse tree produced by apgParser#mask_repeat.
    def enterMask_repeat(self, ctx:apgParser.Mask_repeatContext):
        pass

    # Exit a parse tree produced by apgParser#mask_repeat.
    def exitMask_repeat(self, ctx:apgParser.Mask_repeatContext):
        pass


    # Enter a parse tree produced by apgParser#mask_origin.
    def enterMask_origin(self, ctx:apgParser.Mask_originContext):
        pass

    # Exit a parse tree produced by apgParser#mask_origin.
    def exitMask_origin(self, ctx:apgParser.Mask_originContext):
        pass


    # Enter a parse tree produced by apgParser#mask_clip.
    def enterMask_clip(self, ctx:apgParser.Mask_clipContext):
        pass

    # Exit a parse tree produced by apgParser#mask_clip.
    def exitMask_clip(self, ctx:apgParser.Mask_clipContext):
        pass


    # Enter a parse tree produced by apgParser#date_format.
    def enterDate_format(self, ctx:apgParser.Date_formatContext):
        pass

    # Exit a parse tree produced by apgParser#date_format.
    def exitDate_format(self, ctx:apgParser.Date_formatContext):
        pass


    # Enter a parse tree produced by apgParser#number_format.
    def enterNumber_format(self, ctx:apgParser.Number_formatContext):
        pass

    # Exit a parse tree produced by apgParser#number_format.
    def exitNumber_format(self, ctx:apgParser.Number_formatContext):
        pass


    # Enter a parse tree produced by apgParser#currency_format.
    def enterCurrency_format(self, ctx:apgParser.Currency_formatContext):
        pass

    # Exit a parse tree produced by apgParser#currency_format.
    def exitCurrency_format(self, ctx:apgParser.Currency_formatContext):
        pass


    # Enter a parse tree produced by apgParser#custom_format.
    def enterCustom_format(self, ctx:apgParser.Custom_formatContext):
        pass

    # Exit a parse tree produced by apgParser#custom_format.
    def exitCustom_format(self, ctx:apgParser.Custom_formatContext):
        pass


    # Enter a parse tree produced by apgParser#color_palette.
    def enterColor_palette(self, ctx:apgParser.Color_paletteContext):
        pass

    # Exit a parse tree produced by apgParser#color_palette.
    def exitColor_palette(self, ctx:apgParser.Color_paletteContext):
        pass


    # Enter a parse tree produced by apgParser#color_definition.
    def enterColor_definition(self, ctx:apgParser.Color_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#color_definition.
    def exitColor_definition(self, ctx:apgParser.Color_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#typography_scale.
    def enterTypography_scale(self, ctx:apgParser.Typography_scaleContext):
        pass

    # Exit a parse tree produced by apgParser#typography_scale.
    def exitTypography_scale(self, ctx:apgParser.Typography_scaleContext):
        pass


    # Enter a parse tree produced by apgParser#typography_definition.
    def enterTypography_definition(self, ctx:apgParser.Typography_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#typography_definition.
    def exitTypography_definition(self, ctx:apgParser.Typography_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#font_definition.
    def enterFont_definition(self, ctx:apgParser.Font_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#font_definition.
    def exitFont_definition(self, ctx:apgParser.Font_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#font_property.
    def enterFont_property(self, ctx:apgParser.Font_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#font_property.
    def exitFont_property(self, ctx:apgParser.Font_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#spacing_scale.
    def enterSpacing_scale(self, ctx:apgParser.Spacing_scaleContext):
        pass

    # Exit a parse tree produced by apgParser#spacing_scale.
    def exitSpacing_scale(self, ctx:apgParser.Spacing_scaleContext):
        pass


    # Enter a parse tree produced by apgParser#spacing_definition.
    def enterSpacing_definition(self, ctx:apgParser.Spacing_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#spacing_definition.
    def exitSpacing_definition(self, ctx:apgParser.Spacing_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#shadow_scale.
    def enterShadow_scale(self, ctx:apgParser.Shadow_scaleContext):
        pass

    # Exit a parse tree produced by apgParser#shadow_scale.
    def exitShadow_scale(self, ctx:apgParser.Shadow_scaleContext):
        pass


    # Enter a parse tree produced by apgParser#shadow_definition.
    def enterShadow_definition(self, ctx:apgParser.Shadow_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#shadow_definition.
    def exitShadow_definition(self, ctx:apgParser.Shadow_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#api_endpoint.
    def enterApi_endpoint(self, ctx:apgParser.Api_endpointContext):
        pass

    # Exit a parse tree produced by apgParser#api_endpoint.
    def exitApi_endpoint(self, ctx:apgParser.Api_endpointContext):
        pass


    # Enter a parse tree produced by apgParser#url_string.
    def enterUrl_string(self, ctx:apgParser.Url_stringContext):
        pass

    # Exit a parse tree produced by apgParser#url_string.
    def exitUrl_string(self, ctx:apgParser.Url_stringContext):
        pass


    # Enter a parse tree produced by apgParser#endpoint_config.
    def enterEndpoint_config(self, ctx:apgParser.Endpoint_configContext):
        pass

    # Exit a parse tree produced by apgParser#endpoint_config.
    def exitEndpoint_config(self, ctx:apgParser.Endpoint_configContext):
        pass


    # Enter a parse tree produced by apgParser#endpoint_property.
    def enterEndpoint_property(self, ctx:apgParser.Endpoint_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#endpoint_property.
    def exitEndpoint_property(self, ctx:apgParser.Endpoint_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#header_property.
    def enterHeader_property(self, ctx:apgParser.Header_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#header_property.
    def exitHeader_property(self, ctx:apgParser.Header_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#param_property.
    def enterParam_property(self, ctx:apgParser.Param_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#param_property.
    def exitParam_property(self, ctx:apgParser.Param_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#auth_config.
    def enterAuth_config(self, ctx:apgParser.Auth_configContext):
        pass

    # Exit a parse tree produced by apgParser#auth_config.
    def exitAuth_config(self, ctx:apgParser.Auth_configContext):
        pass


    # Enter a parse tree produced by apgParser#schema_definition.
    def enterSchema_definition(self, ctx:apgParser.Schema_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#schema_definition.
    def exitSchema_definition(self, ctx:apgParser.Schema_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#schema_property.
    def enterSchema_property(self, ctx:apgParser.Schema_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#schema_property.
    def exitSchema_property(self, ctx:apgParser.Schema_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#cascade_rule.
    def enterCascade_rule(self, ctx:apgParser.Cascade_ruleContext):
        pass

    # Exit a parse tree produced by apgParser#cascade_rule.
    def exitCascade_rule(self, ctx:apgParser.Cascade_ruleContext):
        pass


    # Enter a parse tree produced by apgParser#cascade_step.
    def enterCascade_step(self, ctx:apgParser.Cascade_stepContext):
        pass

    # Exit a parse tree produced by apgParser#cascade_step.
    def exitCascade_step(self, ctx:apgParser.Cascade_stepContext):
        pass


    # Enter a parse tree produced by apgParser#condition_expression.
    def enterCondition_expression(self, ctx:apgParser.Condition_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#condition_expression.
    def exitCondition_expression(self, ctx:apgParser.Condition_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#object_property.
    def enterObject_property(self, ctx:apgParser.Object_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#object_property.
    def exitObject_property(self, ctx:apgParser.Object_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#parameter_list.
    def enterParameter_list(self, ctx:apgParser.Parameter_listContext):
        pass

    # Exit a parse tree produced by apgParser#parameter_list.
    def exitParameter_list(self, ctx:apgParser.Parameter_listContext):
        pass


    # Enter a parse tree produced by apgParser#duration_value.
    def enterDuration_value(self, ctx:apgParser.Duration_valueContext):
        pass

    # Exit a parse tree produced by apgParser#duration_value.
    def exitDuration_value(self, ctx:apgParser.Duration_valueContext):
        pass


    # Enter a parse tree produced by apgParser#time_unit.
    def enterTime_unit(self, ctx:apgParser.Time_unitContext):
        pass

    # Exit a parse tree produced by apgParser#time_unit.
    def exitTime_unit(self, ctx:apgParser.Time_unitContext):
        pass


    # Enter a parse tree produced by apgParser#test_definition.
    def enterTest_definition(self, ctx:apgParser.Test_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#test_definition.
    def exitTest_definition(self, ctx:apgParser.Test_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#test_type.
    def enterTest_type(self, ctx:apgParser.Test_typeContext):
        pass

    # Exit a parse tree produced by apgParser#test_type.
    def exitTest_type(self, ctx:apgParser.Test_typeContext):
        pass


    # Enter a parse tree produced by apgParser#test_configuration.
    def enterTest_configuration(self, ctx:apgParser.Test_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#test_configuration.
    def exitTest_configuration(self, ctx:apgParser.Test_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#test_config_item.
    def enterTest_config_item(self, ctx:apgParser.Test_config_itemContext):
        pass

    # Exit a parse tree produced by apgParser#test_config_item.
    def exitTest_config_item(self, ctx:apgParser.Test_config_itemContext):
        pass


    # Enter a parse tree produced by apgParser#browser_specification.
    def enterBrowser_specification(self, ctx:apgParser.Browser_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#browser_specification.
    def exitBrowser_specification(self, ctx:apgParser.Browser_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#browser_config.
    def enterBrowser_config(self, ctx:apgParser.Browser_configContext):
        pass

    # Exit a parse tree produced by apgParser#browser_config.
    def exitBrowser_config(self, ctx:apgParser.Browser_configContext):
        pass


    # Enter a parse tree produced by apgParser#browser_property.
    def enterBrowser_property(self, ctx:apgParser.Browser_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#browser_property.
    def exitBrowser_property(self, ctx:apgParser.Browser_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#device_specification.
    def enterDevice_specification(self, ctx:apgParser.Device_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#device_specification.
    def exitDevice_specification(self, ctx:apgParser.Device_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#device_config.
    def enterDevice_config(self, ctx:apgParser.Device_configContext):
        pass

    # Exit a parse tree produced by apgParser#device_config.
    def exitDevice_config(self, ctx:apgParser.Device_configContext):
        pass


    # Enter a parse tree produced by apgParser#device_property.
    def enterDevice_property(self, ctx:apgParser.Device_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#device_property.
    def exitDevice_property(self, ctx:apgParser.Device_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#test_body.
    def enterTest_body(self, ctx:apgParser.Test_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#test_body.
    def exitTest_body(self, ctx:apgParser.Test_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#test_step.
    def enterTest_step(self, ctx:apgParser.Test_stepContext):
        pass

    # Exit a parse tree produced by apgParser#test_step.
    def exitTest_step(self, ctx:apgParser.Test_stepContext):
        pass


    # Enter a parse tree produced by apgParser#test_action.
    def enterTest_action(self, ctx:apgParser.Test_actionContext):
        pass

    # Exit a parse tree produced by apgParser#test_action.
    def exitTest_action(self, ctx:apgParser.Test_actionContext):
        pass


    # Enter a parse tree produced by apgParser#test_setup.
    def enterTest_setup(self, ctx:apgParser.Test_setupContext):
        pass

    # Exit a parse tree produced by apgParser#test_setup.
    def exitTest_setup(self, ctx:apgParser.Test_setupContext):
        pass


    # Enter a parse tree produced by apgParser#setup_action.
    def enterSetup_action(self, ctx:apgParser.Setup_actionContext):
        pass

    # Exit a parse tree produced by apgParser#setup_action.
    def exitSetup_action(self, ctx:apgParser.Setup_actionContext):
        pass


    # Enter a parse tree produced by apgParser#test_verification.
    def enterTest_verification(self, ctx:apgParser.Test_verificationContext):
        pass

    # Exit a parse tree produced by apgParser#test_verification.
    def exitTest_verification(self, ctx:apgParser.Test_verificationContext):
        pass


    # Enter a parse tree produced by apgParser#verification_condition.
    def enterVerification_condition(self, ctx:apgParser.Verification_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#verification_condition.
    def exitVerification_condition(self, ctx:apgParser.Verification_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#element_verification.
    def enterElement_verification(self, ctx:apgParser.Element_verificationContext):
        pass

    # Exit a parse tree produced by apgParser#element_verification.
    def exitElement_verification(self, ctx:apgParser.Element_verificationContext):
        pass


    # Enter a parse tree produced by apgParser#state_verification.
    def enterState_verification(self, ctx:apgParser.State_verificationContext):
        pass

    # Exit a parse tree produced by apgParser#state_verification.
    def exitState_verification(self, ctx:apgParser.State_verificationContext):
        pass


    # Enter a parse tree produced by apgParser#data_verification.
    def enterData_verification(self, ctx:apgParser.Data_verificationContext):
        pass

    # Exit a parse tree produced by apgParser#data_verification.
    def exitData_verification(self, ctx:apgParser.Data_verificationContext):
        pass


    # Enter a parse tree produced by apgParser#test_assertion.
    def enterTest_assertion(self, ctx:apgParser.Test_assertionContext):
        pass

    # Exit a parse tree produced by apgParser#test_assertion.
    def exitTest_assertion(self, ctx:apgParser.Test_assertionContext):
        pass


    # Enter a parse tree produced by apgParser#assertion_expression.
    def enterAssertion_expression(self, ctx:apgParser.Assertion_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#assertion_expression.
    def exitAssertion_expression(self, ctx:apgParser.Assertion_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#expectation_expression.
    def enterExpectation_expression(self, ctx:apgParser.Expectation_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#expectation_expression.
    def exitExpectation_expression(self, ctx:apgParser.Expectation_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#async_expectation.
    def enterAsync_expectation(self, ctx:apgParser.Async_expectationContext):
        pass

    # Exit a parse tree produced by apgParser#async_expectation.
    def exitAsync_expectation(self, ctx:apgParser.Async_expectationContext):
        pass


    # Enter a parse tree produced by apgParser#verification_expression.
    def enterVerification_expression(self, ctx:apgParser.Verification_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#verification_expression.
    def exitVerification_expression(self, ctx:apgParser.Verification_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#element_state_verification.
    def enterElement_state_verification(self, ctx:apgParser.Element_state_verificationContext):
        pass

    # Exit a parse tree produced by apgParser#element_state_verification.
    def exitElement_state_verification(self, ctx:apgParser.Element_state_verificationContext):
        pass


    # Enter a parse tree produced by apgParser#element_state.
    def enterElement_state(self, ctx:apgParser.Element_stateContext):
        pass

    # Exit a parse tree produced by apgParser#element_state.
    def exitElement_state(self, ctx:apgParser.Element_stateContext):
        pass


    # Enter a parse tree produced by apgParser#element_property.
    def enterElement_property(self, ctx:apgParser.Element_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#element_property.
    def exitElement_property(self, ctx:apgParser.Element_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#exception_type.
    def enterException_type(self, ctx:apgParser.Exception_typeContext):
        pass

    # Exit a parse tree produced by apgParser#exception_type.
    def exitException_type(self, ctx:apgParser.Exception_typeContext):
        pass


    # Enter a parse tree produced by apgParser#selector_expr.
    def enterSelector_expr(self, ctx:apgParser.Selector_exprContext):
        pass

    # Exit a parse tree produced by apgParser#selector_expr.
    def exitSelector_expr(self, ctx:apgParser.Selector_exprContext):
        pass


    # Enter a parse tree produced by apgParser#css_selector.
    def enterCss_selector(self, ctx:apgParser.Css_selectorContext):
        pass

    # Exit a parse tree produced by apgParser#css_selector.
    def exitCss_selector(self, ctx:apgParser.Css_selectorContext):
        pass


    # Enter a parse tree produced by apgParser#xpath_selector.
    def enterXpath_selector(self, ctx:apgParser.Xpath_selectorContext):
        pass

    # Exit a parse tree produced by apgParser#xpath_selector.
    def exitXpath_selector(self, ctx:apgParser.Xpath_selectorContext):
        pass


    # Enter a parse tree produced by apgParser#id_selector.
    def enterId_selector(self, ctx:apgParser.Id_selectorContext):
        pass

    # Exit a parse tree produced by apgParser#id_selector.
    def exitId_selector(self, ctx:apgParser.Id_selectorContext):
        pass


    # Enter a parse tree produced by apgParser#class_selector.
    def enterClass_selector(self, ctx:apgParser.Class_selectorContext):
        pass

    # Exit a parse tree produced by apgParser#class_selector.
    def exitClass_selector(self, ctx:apgParser.Class_selectorContext):
        pass


    # Enter a parse tree produced by apgParser#scroll_options.
    def enterScroll_options(self, ctx:apgParser.Scroll_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#scroll_options.
    def exitScroll_options(self, ctx:apgParser.Scroll_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#scroll_property.
    def enterScroll_property(self, ctx:apgParser.Scroll_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#scroll_property.
    def exitScroll_property(self, ctx:apgParser.Scroll_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#scroll_direction.
    def enterScroll_direction(self, ctx:apgParser.Scroll_directionContext):
        pass

    # Exit a parse tree produced by apgParser#scroll_direction.
    def exitScroll_direction(self, ctx:apgParser.Scroll_directionContext):
        pass


    # Enter a parse tree produced by apgParser#option_expr.
    def enterOption_expr(self, ctx:apgParser.Option_exprContext):
        pass

    # Exit a parse tree produced by apgParser#option_expr.
    def exitOption_expr(self, ctx:apgParser.Option_exprContext):
        pass


    # Enter a parse tree produced by apgParser#code_block.
    def enterCode_block(self, ctx:apgParser.Code_blockContext):
        pass

    # Exit a parse tree produced by apgParser#code_block.
    def exitCode_block(self, ctx:apgParser.Code_blockContext):
        pass


    # Enter a parse tree produced by apgParser#property_matcher.
    def enterProperty_matcher(self, ctx:apgParser.Property_matcherContext):
        pass

    # Exit a parse tree produced by apgParser#property_matcher.
    def exitProperty_matcher(self, ctx:apgParser.Property_matcherContext):
        pass


    # Enter a parse tree produced by apgParser#custom_matcher.
    def enterCustom_matcher(self, ctx:apgParser.Custom_matcherContext):
        pass

    # Exit a parse tree produced by apgParser#custom_matcher.
    def exitCustom_matcher(self, ctx:apgParser.Custom_matcherContext):
        pass


    # Enter a parse tree produced by apgParser#spy_options.
    def enterSpy_options(self, ctx:apgParser.Spy_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#spy_options.
    def exitSpy_options(self, ctx:apgParser.Spy_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#spy_property.
    def enterSpy_property(self, ctx:apgParser.Spy_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#spy_property.
    def exitSpy_property(self, ctx:apgParser.Spy_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#stub_options.
    def enterStub_options(self, ctx:apgParser.Stub_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#stub_options.
    def exitStub_options(self, ctx:apgParser.Stub_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#stub_property.
    def enterStub_property(self, ctx:apgParser.Stub_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#stub_property.
    def exitStub_property(self, ctx:apgParser.Stub_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#fake_options.
    def enterFake_options(self, ctx:apgParser.Fake_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#fake_options.
    def exitFake_options(self, ctx:apgParser.Fake_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#fake_property.
    def enterFake_property(self, ctx:apgParser.Fake_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#fake_property.
    def exitFake_property(self, ctx:apgParser.Fake_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#trace_expression.
    def enterTrace_expression(self, ctx:apgParser.Trace_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#trace_expression.
    def exitTrace_expression(self, ctx:apgParser.Trace_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#profile_expression.
    def enterProfile_expression(self, ctx:apgParser.Profile_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#profile_expression.
    def exitProfile_expression(self, ctx:apgParser.Profile_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#benchmark_expression.
    def enterBenchmark_expression(self, ctx:apgParser.Benchmark_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#benchmark_expression.
    def exitBenchmark_expression(self, ctx:apgParser.Benchmark_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#vault_specification.
    def enterVault_specification(self, ctx:apgParser.Vault_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#vault_specification.
    def exitVault_specification(self, ctx:apgParser.Vault_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#vault_provider.
    def enterVault_provider(self, ctx:apgParser.Vault_providerContext):
        pass

    # Exit a parse tree produced by apgParser#vault_provider.
    def exitVault_provider(self, ctx:apgParser.Vault_providerContext):
        pass


    # Enter a parse tree produced by apgParser#vault_config.
    def enterVault_config(self, ctx:apgParser.Vault_configContext):
        pass

    # Exit a parse tree produced by apgParser#vault_config.
    def exitVault_config(self, ctx:apgParser.Vault_configContext):
        pass


    # Enter a parse tree produced by apgParser#vault_property.
    def enterVault_property(self, ctx:apgParser.Vault_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#vault_property.
    def exitVault_property(self, ctx:apgParser.Vault_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#database_specification.
    def enterDatabase_specification(self, ctx:apgParser.Database_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#database_specification.
    def exitDatabase_specification(self, ctx:apgParser.Database_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#database_type.
    def enterDatabase_type(self, ctx:apgParser.Database_typeContext):
        pass

    # Exit a parse tree produced by apgParser#database_type.
    def exitDatabase_type(self, ctx:apgParser.Database_typeContext):
        pass


    # Enter a parse tree produced by apgParser#database_config.
    def enterDatabase_config(self, ctx:apgParser.Database_configContext):
        pass

    # Exit a parse tree produced by apgParser#database_config.
    def exitDatabase_config(self, ctx:apgParser.Database_configContext):
        pass


    # Enter a parse tree produced by apgParser#database_property.
    def enterDatabase_property(self, ctx:apgParser.Database_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#database_property.
    def exitDatabase_property(self, ctx:apgParser.Database_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#database_schema.
    def enterDatabase_schema(self, ctx:apgParser.Database_schemaContext):
        pass

    # Exit a parse tree produced by apgParser#database_schema.
    def exitDatabase_schema(self, ctx:apgParser.Database_schemaContext):
        pass


    # Enter a parse tree produced by apgParser#enum_variant_decl.
    def enterEnum_variant_decl(self, ctx:apgParser.Enum_variant_declContext):
        pass

    # Exit a parse tree produced by apgParser#enum_variant_decl.
    def exitEnum_variant_decl(self, ctx:apgParser.Enum_variant_declContext):
        pass


    # Enter a parse tree produced by apgParser#enum_variant_doc.
    def enterEnum_variant_doc(self, ctx:apgParser.Enum_variant_docContext):
        pass

    # Exit a parse tree produced by apgParser#enum_variant_doc.
    def exitEnum_variant_doc(self, ctx:apgParser.Enum_variant_docContext):
        pass


    # Enter a parse tree produced by apgParser#state_transition.
    def enterState_transition(self, ctx:apgParser.State_transitionContext):
        pass

    # Exit a parse tree produced by apgParser#state_transition.
    def exitState_transition(self, ctx:apgParser.State_transitionContext):
        pass


    # Enter a parse tree produced by apgParser#state_transition_props.
    def enterState_transition_props(self, ctx:apgParser.State_transition_propsContext):
        pass

    # Exit a parse tree produced by apgParser#state_transition_props.
    def exitState_transition_props(self, ctx:apgParser.State_transition_propsContext):
        pass


    # Enter a parse tree produced by apgParser#state_transition_prop.
    def enterState_transition_prop(self, ctx:apgParser.State_transition_propContext):
        pass

    # Exit a parse tree produced by apgParser#state_transition_prop.
    def exitState_transition_prop(self, ctx:apgParser.State_transition_propContext):
        pass


    # Enter a parse tree produced by apgParser#schema_element.
    def enterSchema_element(self, ctx:apgParser.Schema_elementContext):
        pass

    # Exit a parse tree produced by apgParser#schema_element.
    def exitSchema_element(self, ctx:apgParser.Schema_elementContext):
        pass


    # Enter a parse tree produced by apgParser#table_definition.
    def enterTable_definition(self, ctx:apgParser.Table_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#table_definition.
    def exitTable_definition(self, ctx:apgParser.Table_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#table_alias.
    def enterTable_alias(self, ctx:apgParser.Table_aliasContext):
        pass

    # Exit a parse tree produced by apgParser#table_alias.
    def exitTable_alias(self, ctx:apgParser.Table_aliasContext):
        pass


    # Enter a parse tree produced by apgParser#table_note.
    def enterTable_note(self, ctx:apgParser.Table_noteContext):
        pass

    # Exit a parse tree produced by apgParser#table_note.
    def exitTable_note(self, ctx:apgParser.Table_noteContext):
        pass


    # Enter a parse tree produced by apgParser#table_body.
    def enterTable_body(self, ctx:apgParser.Table_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#table_body.
    def exitTable_body(self, ctx:apgParser.Table_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#column_definition.
    def enterColumn_definition(self, ctx:apgParser.Column_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#column_definition.
    def exitColumn_definition(self, ctx:apgParser.Column_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#column_type.
    def enterColumn_type(self, ctx:apgParser.Column_typeContext):
        pass

    # Exit a parse tree produced by apgParser#column_type.
    def exitColumn_type(self, ctx:apgParser.Column_typeContext):
        pass


    # Enter a parse tree produced by apgParser#db_data_type.
    def enterDb_data_type(self, ctx:apgParser.Db_data_typeContext):
        pass

    # Exit a parse tree produced by apgParser#db_data_type.
    def exitDb_data_type(self, ctx:apgParser.Db_data_typeContext):
        pass


    # Enter a parse tree produced by apgParser#column_nullable.
    def enterColumn_nullable(self, ctx:apgParser.Column_nullableContext):
        pass

    # Exit a parse tree produced by apgParser#column_nullable.
    def exitColumn_nullable(self, ctx:apgParser.Column_nullableContext):
        pass


    # Enter a parse tree produced by apgParser#column_constraints.
    def enterColumn_constraints(self, ctx:apgParser.Column_constraintsContext):
        pass

    # Exit a parse tree produced by apgParser#column_constraints.
    def exitColumn_constraints(self, ctx:apgParser.Column_constraintsContext):
        pass


    # Enter a parse tree produced by apgParser#column_constraint.
    def enterColumn_constraint(self, ctx:apgParser.Column_constraintContext):
        pass

    # Exit a parse tree produced by apgParser#column_constraint.
    def exitColumn_constraint(self, ctx:apgParser.Column_constraintContext):
        pass


    # Enter a parse tree produced by apgParser#reference_spec.
    def enterReference_spec(self, ctx:apgParser.Reference_specContext):
        pass

    # Exit a parse tree produced by apgParser#reference_spec.
    def exitReference_spec(self, ctx:apgParser.Reference_specContext):
        pass


    # Enter a parse tree produced by apgParser#reference_type.
    def enterReference_type(self, ctx:apgParser.Reference_typeContext):
        pass

    # Exit a parse tree produced by apgParser#reference_type.
    def exitReference_type(self, ctx:apgParser.Reference_typeContext):
        pass


    # Enter a parse tree produced by apgParser#table_column_ref.
    def enterTable_column_ref(self, ctx:apgParser.Table_column_refContext):
        pass

    # Exit a parse tree produced by apgParser#table_column_ref.
    def exitTable_column_ref(self, ctx:apgParser.Table_column_refContext):
        pass


    # Enter a parse tree produced by apgParser#table_reference.
    def enterTable_reference(self, ctx:apgParser.Table_referenceContext):
        pass

    # Exit a parse tree produced by apgParser#table_reference.
    def exitTable_reference(self, ctx:apgParser.Table_referenceContext):
        pass


    # Enter a parse tree produced by apgParser#reference_options.
    def enterReference_options(self, ctx:apgParser.Reference_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#reference_options.
    def exitReference_options(self, ctx:apgParser.Reference_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#reference_option.
    def enterReference_option(self, ctx:apgParser.Reference_optionContext):
        pass

    # Exit a parse tree produced by apgParser#reference_option.
    def exitReference_option(self, ctx:apgParser.Reference_optionContext):
        pass


    # Enter a parse tree produced by apgParser#reference_action.
    def enterReference_action(self, ctx:apgParser.Reference_actionContext):
        pass

    # Exit a parse tree produced by apgParser#reference_action.
    def exitReference_action(self, ctx:apgParser.Reference_actionContext):
        pass


    # Enter a parse tree produced by apgParser#table_index.
    def enterTable_index(self, ctx:apgParser.Table_indexContext):
        pass

    # Exit a parse tree produced by apgParser#table_index.
    def exitTable_index(self, ctx:apgParser.Table_indexContext):
        pass


    # Enter a parse tree produced by apgParser#index_definition.
    def enterIndex_definition(self, ctx:apgParser.Index_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#index_definition.
    def exitIndex_definition(self, ctx:apgParser.Index_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#index_columns.
    def enterIndex_columns(self, ctx:apgParser.Index_columnsContext):
        pass

    # Exit a parse tree produced by apgParser#index_columns.
    def exitIndex_columns(self, ctx:apgParser.Index_columnsContext):
        pass


    # Enter a parse tree produced by apgParser#index_options.
    def enterIndex_options(self, ctx:apgParser.Index_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#index_options.
    def exitIndex_options(self, ctx:apgParser.Index_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#index_option.
    def enterIndex_option(self, ctx:apgParser.Index_optionContext):
        pass

    # Exit a parse tree produced by apgParser#index_option.
    def exitIndex_option(self, ctx:apgParser.Index_optionContext):
        pass


    # Enter a parse tree produced by apgParser#index_type.
    def enterIndex_type(self, ctx:apgParser.Index_typeContext):
        pass

    # Exit a parse tree produced by apgParser#index_type.
    def exitIndex_type(self, ctx:apgParser.Index_typeContext):
        pass


    # Enter a parse tree produced by apgParser#table_constraint.
    def enterTable_constraint(self, ctx:apgParser.Table_constraintContext):
        pass

    # Exit a parse tree produced by apgParser#table_constraint.
    def exitTable_constraint(self, ctx:apgParser.Table_constraintContext):
        pass


    # Enter a parse tree produced by apgParser#constraint_definition.
    def enterConstraint_definition(self, ctx:apgParser.Constraint_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#constraint_definition.
    def exitConstraint_definition(self, ctx:apgParser.Constraint_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#enum_definition.
    def enterEnum_definition(self, ctx:apgParser.Enum_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#enum_definition.
    def exitEnum_definition(self, ctx:apgParser.Enum_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#enum_values.
    def enterEnum_values(self, ctx:apgParser.Enum_valuesContext):
        pass

    # Exit a parse tree produced by apgParser#enum_values.
    def exitEnum_values(self, ctx:apgParser.Enum_valuesContext):
        pass


    # Enter a parse tree produced by apgParser#enum_value.
    def enterEnum_value(self, ctx:apgParser.Enum_valueContext):
        pass

    # Exit a parse tree produced by apgParser#enum_value.
    def exitEnum_value(self, ctx:apgParser.Enum_valueContext):
        pass


    # Enter a parse tree produced by apgParser#enum_note.
    def enterEnum_note(self, ctx:apgParser.Enum_noteContext):
        pass

    # Exit a parse tree produced by apgParser#enum_note.
    def exitEnum_note(self, ctx:apgParser.Enum_noteContext):
        pass


    # Enter a parse tree produced by apgParser#table_group.
    def enterTable_group(self, ctx:apgParser.Table_groupContext):
        pass

    # Exit a parse tree produced by apgParser#table_group.
    def exitTable_group(self, ctx:apgParser.Table_groupContext):
        pass


    # Enter a parse tree produced by apgParser#note_definition.
    def enterNote_definition(self, ctx:apgParser.Note_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#note_definition.
    def exitNote_definition(self, ctx:apgParser.Note_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_definition.
    def enterTrigger_definition(self, ctx:apgParser.Trigger_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_definition.
    def exitTrigger_definition(self, ctx:apgParser.Trigger_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_spec.
    def enterTrigger_spec(self, ctx:apgParser.Trigger_specContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_spec.
    def exitTrigger_spec(self, ctx:apgParser.Trigger_specContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_timing.
    def enterTrigger_timing(self, ctx:apgParser.Trigger_timingContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_timing.
    def exitTrigger_timing(self, ctx:apgParser.Trigger_timingContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_event.
    def enterTrigger_event(self, ctx:apgParser.Trigger_eventContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_event.
    def exitTrigger_event(self, ctx:apgParser.Trigger_eventContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_condition.
    def enterTrigger_condition(self, ctx:apgParser.Trigger_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_condition.
    def exitTrigger_condition(self, ctx:apgParser.Trigger_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_level.
    def enterTrigger_level(self, ctx:apgParser.Trigger_levelContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_level.
    def exitTrigger_level(self, ctx:apgParser.Trigger_levelContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_body.
    def enterTrigger_body(self, ctx:apgParser.Trigger_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_body.
    def exitTrigger_body(self, ctx:apgParser.Trigger_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#trigger_statement.
    def enterTrigger_statement(self, ctx:apgParser.Trigger_statementContext):
        pass

    # Exit a parse tree produced by apgParser#trigger_statement.
    def exitTrigger_statement(self, ctx:apgParser.Trigger_statementContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_definition.
    def enterProcedure_definition(self, ctx:apgParser.Procedure_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_definition.
    def exitProcedure_definition(self, ctx:apgParser.Procedure_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_parameters.
    def enterProcedure_parameters(self, ctx:apgParser.Procedure_parametersContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_parameters.
    def exitProcedure_parameters(self, ctx:apgParser.Procedure_parametersContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_parameter.
    def enterProcedure_parameter(self, ctx:apgParser.Procedure_parameterContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_parameter.
    def exitProcedure_parameter(self, ctx:apgParser.Procedure_parameterContext):
        pass


    # Enter a parse tree produced by apgParser#parameter_mode.
    def enterParameter_mode(self, ctx:apgParser.Parameter_modeContext):
        pass

    # Exit a parse tree produced by apgParser#parameter_mode.
    def exitParameter_mode(self, ctx:apgParser.Parameter_modeContext):
        pass


    # Enter a parse tree produced by apgParser#default_value.
    def enterDefault_value(self, ctx:apgParser.Default_valueContext):
        pass

    # Exit a parse tree produced by apgParser#default_value.
    def exitDefault_value(self, ctx:apgParser.Default_valueContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_options.
    def enterProcedure_options(self, ctx:apgParser.Procedure_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_options.
    def exitProcedure_options(self, ctx:apgParser.Procedure_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_option.
    def enterProcedure_option(self, ctx:apgParser.Procedure_optionContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_option.
    def exitProcedure_option(self, ctx:apgParser.Procedure_optionContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_language.
    def enterProcedure_language(self, ctx:apgParser.Procedure_languageContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_language.
    def exitProcedure_language(self, ctx:apgParser.Procedure_languageContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_body.
    def enterProcedure_body(self, ctx:apgParser.Procedure_bodyContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_body.
    def exitProcedure_body(self, ctx:apgParser.Procedure_bodyContext):
        pass


    # Enter a parse tree produced by apgParser#procedure_statement.
    def enterProcedure_statement(self, ctx:apgParser.Procedure_statementContext):
        pass

    # Exit a parse tree produced by apgParser#procedure_statement.
    def exitProcedure_statement(self, ctx:apgParser.Procedure_statementContext):
        pass


    # Enter a parse tree produced by apgParser#function_definition.
    def enterFunction_definition(self, ctx:apgParser.Function_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#function_definition.
    def exitFunction_definition(self, ctx:apgParser.Function_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#db_return_type.
    def enterDb_return_type(self, ctx:apgParser.Db_return_typeContext):
        pass

    # Exit a parse tree produced by apgParser#db_return_type.
    def exitDb_return_type(self, ctx:apgParser.Db_return_typeContext):
        pass


    # Enter a parse tree produced by apgParser#table_column_list.
    def enterTable_column_list(self, ctx:apgParser.Table_column_listContext):
        pass

    # Exit a parse tree produced by apgParser#table_column_list.
    def exitTable_column_list(self, ctx:apgParser.Table_column_listContext):
        pass


    # Enter a parse tree produced by apgParser#table_column_def.
    def enterTable_column_def(self, ctx:apgParser.Table_column_defContext):
        pass

    # Exit a parse tree produced by apgParser#table_column_def.
    def exitTable_column_def(self, ctx:apgParser.Table_column_defContext):
        pass


    # Enter a parse tree produced by apgParser#function_options.
    def enterFunction_options(self, ctx:apgParser.Function_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#function_options.
    def exitFunction_options(self, ctx:apgParser.Function_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#function_option.
    def enterFunction_option(self, ctx:apgParser.Function_optionContext):
        pass

    # Exit a parse tree produced by apgParser#function_option.
    def exitFunction_option(self, ctx:apgParser.Function_optionContext):
        pass


    # Enter a parse tree produced by apgParser#view_definition.
    def enterView_definition(self, ctx:apgParser.View_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#view_definition.
    def exitView_definition(self, ctx:apgParser.View_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#view_options.
    def enterView_options(self, ctx:apgParser.View_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#view_options.
    def exitView_options(self, ctx:apgParser.View_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#view_option.
    def enterView_option(self, ctx:apgParser.View_optionContext):
        pass

    # Exit a parse tree produced by apgParser#view_option.
    def exitView_option(self, ctx:apgParser.View_optionContext):
        pass


    # Enter a parse tree produced by apgParser#sql_statement.
    def enterSql_statement(self, ctx:apgParser.Sql_statementContext):
        pass

    # Exit a parse tree produced by apgParser#sql_statement.
    def exitSql_statement(self, ctx:apgParser.Sql_statementContext):
        pass


    # Enter a parse tree produced by apgParser#sql_query.
    def enterSql_query(self, ctx:apgParser.Sql_queryContext):
        pass

    # Exit a parse tree produced by apgParser#sql_query.
    def exitSql_query(self, ctx:apgParser.Sql_queryContext):
        pass


    # Enter a parse tree produced by apgParser#select_statement.
    def enterSelect_statement(self, ctx:apgParser.Select_statementContext):
        pass

    # Exit a parse tree produced by apgParser#select_statement.
    def exitSelect_statement(self, ctx:apgParser.Select_statementContext):
        pass


    # Enter a parse tree produced by apgParser#select_list.
    def enterSelect_list(self, ctx:apgParser.Select_listContext):
        pass

    # Exit a parse tree produced by apgParser#select_list.
    def exitSelect_list(self, ctx:apgParser.Select_listContext):
        pass


    # Enter a parse tree produced by apgParser#select_item.
    def enterSelect_item(self, ctx:apgParser.Select_itemContext):
        pass

    # Exit a parse tree produced by apgParser#select_item.
    def exitSelect_item(self, ctx:apgParser.Select_itemContext):
        pass


    # Enter a parse tree produced by apgParser#from_clause.
    def enterFrom_clause(self, ctx:apgParser.From_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#from_clause.
    def exitFrom_clause(self, ctx:apgParser.From_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#where_clause.
    def enterWhere_clause(self, ctx:apgParser.Where_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#where_clause.
    def exitWhere_clause(self, ctx:apgParser.Where_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#group_by_clause.
    def enterGroup_by_clause(self, ctx:apgParser.Group_by_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#group_by_clause.
    def exitGroup_by_clause(self, ctx:apgParser.Group_by_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#having_clause.
    def enterHaving_clause(self, ctx:apgParser.Having_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#having_clause.
    def exitHaving_clause(self, ctx:apgParser.Having_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#order_by_clause.
    def enterOrder_by_clause(self, ctx:apgParser.Order_by_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#order_by_clause.
    def exitOrder_by_clause(self, ctx:apgParser.Order_by_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#order_item.
    def enterOrder_item(self, ctx:apgParser.Order_itemContext):
        pass

    # Exit a parse tree produced by apgParser#order_item.
    def exitOrder_item(self, ctx:apgParser.Order_itemContext):
        pass


    # Enter a parse tree produced by apgParser#limit_clause.
    def enterLimit_clause(self, ctx:apgParser.Limit_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#limit_clause.
    def exitLimit_clause(self, ctx:apgParser.Limit_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#insert_statement.
    def enterInsert_statement(self, ctx:apgParser.Insert_statementContext):
        pass

    # Exit a parse tree produced by apgParser#insert_statement.
    def exitInsert_statement(self, ctx:apgParser.Insert_statementContext):
        pass


    # Enter a parse tree produced by apgParser#update_statement.
    def enterUpdate_statement(self, ctx:apgParser.Update_statementContext):
        pass

    # Exit a parse tree produced by apgParser#update_statement.
    def exitUpdate_statement(self, ctx:apgParser.Update_statementContext):
        pass


    # Enter a parse tree produced by apgParser#delete_statement.
    def enterDelete_statement(self, ctx:apgParser.Delete_statementContext):
        pass

    # Exit a parse tree produced by apgParser#delete_statement.
    def exitDelete_statement(self, ctx:apgParser.Delete_statementContext):
        pass


    # Enter a parse tree produced by apgParser#execute_statement.
    def enterExecute_statement(self, ctx:apgParser.Execute_statementContext):
        pass

    # Exit a parse tree produced by apgParser#execute_statement.
    def exitExecute_statement(self, ctx:apgParser.Execute_statementContext):
        pass


    # Enter a parse tree produced by apgParser#column_list.
    def enterColumn_list(self, ctx:apgParser.Column_listContext):
        pass

    # Exit a parse tree produced by apgParser#column_list.
    def exitColumn_list(self, ctx:apgParser.Column_listContext):
        pass


    # Enter a parse tree produced by apgParser#value_list.
    def enterValue_list(self, ctx:apgParser.Value_listContext):
        pass

    # Exit a parse tree produced by apgParser#value_list.
    def exitValue_list(self, ctx:apgParser.Value_listContext):
        pass


    # Enter a parse tree produced by apgParser#assignment_list.
    def enterAssignment_list(self, ctx:apgParser.Assignment_listContext):
        pass

    # Exit a parse tree produced by apgParser#assignment_list.
    def exitAssignment_list(self, ctx:apgParser.Assignment_listContext):
        pass


    # Enter a parse tree produced by apgParser#db_assignment.
    def enterDb_assignment(self, ctx:apgParser.Db_assignmentContext):
        pass

    # Exit a parse tree produced by apgParser#db_assignment.
    def exitDb_assignment(self, ctx:apgParser.Db_assignmentContext):
        pass


    # Enter a parse tree produced by apgParser#control_flow_statement.
    def enterControl_flow_statement(self, ctx:apgParser.Control_flow_statementContext):
        pass

    # Exit a parse tree produced by apgParser#control_flow_statement.
    def exitControl_flow_statement(self, ctx:apgParser.Control_flow_statementContext):
        pass


    # Enter a parse tree produced by apgParser#loop_statement.
    def enterLoop_statement(self, ctx:apgParser.Loop_statementContext):
        pass

    # Exit a parse tree produced by apgParser#loop_statement.
    def exitLoop_statement(self, ctx:apgParser.Loop_statementContext):
        pass


    # Enter a parse tree produced by apgParser#case_statement.
    def enterCase_statement(self, ctx:apgParser.Case_statementContext):
        pass

    # Exit a parse tree produced by apgParser#case_statement.
    def exitCase_statement(self, ctx:apgParser.Case_statementContext):
        pass


    # Enter a parse tree produced by apgParser#db_when_clause.
    def enterDb_when_clause(self, ctx:apgParser.Db_when_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#db_when_clause.
    def exitDb_when_clause(self, ctx:apgParser.Db_when_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#db_else_clause.
    def enterDb_else_clause(self, ctx:apgParser.Db_else_clauseContext):
        pass

    # Exit a parse tree produced by apgParser#db_else_clause.
    def exitDb_else_clause(self, ctx:apgParser.Db_else_clauseContext):
        pass


    # Enter a parse tree produced by apgParser#exception_handling.
    def enterException_handling(self, ctx:apgParser.Exception_handlingContext):
        pass

    # Exit a parse tree produced by apgParser#exception_handling.
    def exitException_handling(self, ctx:apgParser.Exception_handlingContext):
        pass


    # Enter a parse tree produced by apgParser#exception_condition.
    def enterException_condition(self, ctx:apgParser.Exception_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#exception_condition.
    def exitException_condition(self, ctx:apgParser.Exception_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#db_return_statement.
    def enterDb_return_statement(self, ctx:apgParser.Db_return_statementContext):
        pass

    # Exit a parse tree produced by apgParser#db_return_statement.
    def exitDb_return_statement(self, ctx:apgParser.Db_return_statementContext):
        pass


    # Enter a parse tree produced by apgParser#vector_index_definition.
    def enterVector_index_definition(self, ctx:apgParser.Vector_index_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#vector_index_definition.
    def exitVector_index_definition(self, ctx:apgParser.Vector_index_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#vector_index_options.
    def enterVector_index_options(self, ctx:apgParser.Vector_index_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#vector_index_options.
    def exitVector_index_options(self, ctx:apgParser.Vector_index_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#vector_index_option.
    def enterVector_index_option(self, ctx:apgParser.Vector_index_optionContext):
        pass

    # Exit a parse tree produced by apgParser#vector_index_option.
    def exitVector_index_option(self, ctx:apgParser.Vector_index_optionContext):
        pass


    # Enter a parse tree produced by apgParser#vector_index_method.
    def enterVector_index_method(self, ctx:apgParser.Vector_index_methodContext):
        pass

    # Exit a parse tree produced by apgParser#vector_index_method.
    def exitVector_index_method(self, ctx:apgParser.Vector_index_methodContext):
        pass


    # Enter a parse tree produced by apgParser#distance_function.
    def enterDistance_function(self, ctx:apgParser.Distance_functionContext):
        pass

    # Exit a parse tree produced by apgParser#distance_function.
    def exitDistance_function(self, ctx:apgParser.Distance_functionContext):
        pass


    # Enter a parse tree produced by apgParser#vector_constraint.
    def enterVector_constraint(self, ctx:apgParser.Vector_constraintContext):
        pass

    # Exit a parse tree produced by apgParser#vector_constraint.
    def exitVector_constraint(self, ctx:apgParser.Vector_constraintContext):
        pass


    # Enter a parse tree produced by apgParser#vector_constraint_expression.
    def enterVector_constraint_expression(self, ctx:apgParser.Vector_constraint_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#vector_constraint_expression.
    def exitVector_constraint_expression(self, ctx:apgParser.Vector_constraint_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#vector_column_constraint.
    def enterVector_column_constraint(self, ctx:apgParser.Vector_column_constraintContext):
        pass

    # Exit a parse tree produced by apgParser#vector_column_constraint.
    def exitVector_column_constraint(self, ctx:apgParser.Vector_column_constraintContext):
        pass


    # Enter a parse tree produced by apgParser#apg_statement.
    def enterApg_statement(self, ctx:apgParser.Apg_statementContext):
        pass

    # Exit a parse tree produced by apgParser#apg_statement.
    def exitApg_statement(self, ctx:apgParser.Apg_statementContext):
        pass


    # Enter a parse tree produced by apgParser#sql_expression.
    def enterSql_expression(self, ctx:apgParser.Sql_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#sql_expression.
    def exitSql_expression(self, ctx:apgParser.Sql_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#db_parameter_list.
    def enterDb_parameter_list(self, ctx:apgParser.Db_parameter_listContext):
        pass

    # Exit a parse tree produced by apgParser#db_parameter_list.
    def exitDb_parameter_list(self, ctx:apgParser.Db_parameter_listContext):
        pass


    # Enter a parse tree produced by apgParser#db_statement_block.
    def enterDb_statement_block(self, ctx:apgParser.Db_statement_blockContext):
        pass

    # Exit a parse tree produced by apgParser#db_statement_block.
    def exitDb_statement_block(self, ctx:apgParser.Db_statement_blockContext):
        pass


    # Enter a parse tree produced by apgParser#matcher_expression.
    def enterMatcher_expression(self, ctx:apgParser.Matcher_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#matcher_expression.
    def exitMatcher_expression(self, ctx:apgParser.Matcher_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#mock_configuration.
    def enterMock_configuration(self, ctx:apgParser.Mock_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#mock_configuration.
    def exitMock_configuration(self, ctx:apgParser.Mock_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#mock_item.
    def enterMock_item(self, ctx:apgParser.Mock_itemContext):
        pass

    # Exit a parse tree produced by apgParser#mock_item.
    def exitMock_item(self, ctx:apgParser.Mock_itemContext):
        pass


    # Enter a parse tree produced by apgParser#mock_specification.
    def enterMock_specification(self, ctx:apgParser.Mock_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#mock_specification.
    def exitMock_specification(self, ctx:apgParser.Mock_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#mock_options.
    def enterMock_options(self, ctx:apgParser.Mock_optionsContext):
        pass

    # Exit a parse tree produced by apgParser#mock_options.
    def exitMock_options(self, ctx:apgParser.Mock_optionsContext):
        pass


    # Enter a parse tree produced by apgParser#mock_option.
    def enterMock_option(self, ctx:apgParser.Mock_optionContext):
        pass

    # Exit a parse tree produced by apgParser#mock_option.
    def exitMock_option(self, ctx:apgParser.Mock_optionContext):
        pass


    # Enter a parse tree produced by apgParser#debug_statement.
    def enterDebug_statement(self, ctx:apgParser.Debug_statementContext):
        pass

    # Exit a parse tree produced by apgParser#debug_statement.
    def exitDebug_statement(self, ctx:apgParser.Debug_statementContext):
        pass


    # Enter a parse tree produced by apgParser#debug_expression.
    def enterDebug_expression(self, ctx:apgParser.Debug_expressionContext):
        pass

    # Exit a parse tree produced by apgParser#debug_expression.
    def exitDebug_expression(self, ctx:apgParser.Debug_expressionContext):
        pass


    # Enter a parse tree produced by apgParser#debug_info.
    def enterDebug_info(self, ctx:apgParser.Debug_infoContext):
        pass

    # Exit a parse tree produced by apgParser#debug_info.
    def exitDebug_info(self, ctx:apgParser.Debug_infoContext):
        pass


    # Enter a parse tree produced by apgParser#breakpoint_condition.
    def enterBreakpoint_condition(self, ctx:apgParser.Breakpoint_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#breakpoint_condition.
    def exitBreakpoint_condition(self, ctx:apgParser.Breakpoint_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#config_definition.
    def enterConfig_definition(self, ctx:apgParser.Config_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#config_definition.
    def exitConfig_definition(self, ctx:apgParser.Config_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#config_scope.
    def enterConfig_scope(self, ctx:apgParser.Config_scopeContext):
        pass

    # Exit a parse tree produced by apgParser#config_scope.
    def exitConfig_scope(self, ctx:apgParser.Config_scopeContext):
        pass


    # Enter a parse tree produced by apgParser#config_source.
    def enterConfig_source(self, ctx:apgParser.Config_sourceContext):
        pass

    # Exit a parse tree produced by apgParser#config_source.
    def exitConfig_source(self, ctx:apgParser.Config_sourceContext):
        pass


    # Enter a parse tree produced by apgParser#config_validation.
    def enterConfig_validation(self, ctx:apgParser.Config_validationContext):
        pass

    # Exit a parse tree produced by apgParser#config_validation.
    def exitConfig_validation(self, ctx:apgParser.Config_validationContext):
        pass


    # Enter a parse tree produced by apgParser#secret_definition.
    def enterSecret_definition(self, ctx:apgParser.Secret_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#secret_definition.
    def exitSecret_definition(self, ctx:apgParser.Secret_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#notification_definition.
    def enterNotification_definition(self, ctx:apgParser.Notification_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#notification_definition.
    def exitNotification_definition(self, ctx:apgParser.Notification_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#notification_type.
    def enterNotification_type(self, ctx:apgParser.Notification_typeContext):
        pass

    # Exit a parse tree produced by apgParser#notification_type.
    def exitNotification_type(self, ctx:apgParser.Notification_typeContext):
        pass


    # Enter a parse tree produced by apgParser#notification_configuration.
    def enterNotification_configuration(self, ctx:apgParser.Notification_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#notification_configuration.
    def exitNotification_configuration(self, ctx:apgParser.Notification_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#notification_property.
    def enterNotification_property(self, ctx:apgParser.Notification_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#notification_property.
    def exitNotification_property(self, ctx:apgParser.Notification_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#recipient_list.
    def enterRecipient_list(self, ctx:apgParser.Recipient_listContext):
        pass

    # Exit a parse tree produced by apgParser#recipient_list.
    def exitRecipient_list(self, ctx:apgParser.Recipient_listContext):
        pass


    # Enter a parse tree produced by apgParser#recipient.
    def enterRecipient(self, ctx:apgParser.RecipientContext):
        pass

    # Exit a parse tree produced by apgParser#recipient.
    def exitRecipient(self, ctx:apgParser.RecipientContext):
        pass


    # Enter a parse tree produced by apgParser#recipient_group.
    def enterRecipient_group(self, ctx:apgParser.Recipient_groupContext):
        pass

    # Exit a parse tree produced by apgParser#recipient_group.
    def exitRecipient_group(self, ctx:apgParser.Recipient_groupContext):
        pass


    # Enter a parse tree produced by apgParser#dynamic_recipient_list.
    def enterDynamic_recipient_list(self, ctx:apgParser.Dynamic_recipient_listContext):
        pass

    # Exit a parse tree produced by apgParser#dynamic_recipient_list.
    def exitDynamic_recipient_list(self, ctx:apgParser.Dynamic_recipient_listContext):
        pass


    # Enter a parse tree produced by apgParser#priority_level.
    def enterPriority_level(self, ctx:apgParser.Priority_levelContext):
        pass

    # Exit a parse tree produced by apgParser#priority_level.
    def exitPriority_level(self, ctx:apgParser.Priority_levelContext):
        pass


    # Enter a parse tree produced by apgParser#alert_definition.
    def enterAlert_definition(self, ctx:apgParser.Alert_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#alert_definition.
    def exitAlert_definition(self, ctx:apgParser.Alert_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#alert_type.
    def enterAlert_type(self, ctx:apgParser.Alert_typeContext):
        pass

    # Exit a parse tree produced by apgParser#alert_type.
    def exitAlert_type(self, ctx:apgParser.Alert_typeContext):
        pass


    # Enter a parse tree produced by apgParser#alert_configuration.
    def enterAlert_configuration(self, ctx:apgParser.Alert_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#alert_configuration.
    def exitAlert_configuration(self, ctx:apgParser.Alert_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#alert_property.
    def enterAlert_property(self, ctx:apgParser.Alert_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#alert_property.
    def exitAlert_property(self, ctx:apgParser.Alert_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#alert_condition.
    def enterAlert_condition(self, ctx:apgParser.Alert_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#alert_condition.
    def exitAlert_condition(self, ctx:apgParser.Alert_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#threshold_condition.
    def enterThreshold_condition(self, ctx:apgParser.Threshold_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#threshold_condition.
    def exitThreshold_condition(self, ctx:apgParser.Threshold_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#anomaly_condition.
    def enterAnomaly_condition(self, ctx:apgParser.Anomaly_conditionContext):
        pass

    # Exit a parse tree produced by apgParser#anomaly_condition.
    def exitAnomaly_condition(self, ctx:apgParser.Anomaly_conditionContext):
        pass


    # Enter a parse tree produced by apgParser#severity_level.
    def enterSeverity_level(self, ctx:apgParser.Severity_levelContext):
        pass

    # Exit a parse tree produced by apgParser#severity_level.
    def exitSeverity_level(self, ctx:apgParser.Severity_levelContext):
        pass


    # Enter a parse tree produced by apgParser#logger_definition.
    def enterLogger_definition(self, ctx:apgParser.Logger_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#logger_definition.
    def exitLogger_definition(self, ctx:apgParser.Logger_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#logger_type.
    def enterLogger_type(self, ctx:apgParser.Logger_typeContext):
        pass

    # Exit a parse tree produced by apgParser#logger_type.
    def exitLogger_type(self, ctx:apgParser.Logger_typeContext):
        pass


    # Enter a parse tree produced by apgParser#logger_configuration.
    def enterLogger_configuration(self, ctx:apgParser.Logger_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#logger_configuration.
    def exitLogger_configuration(self, ctx:apgParser.Logger_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#logger_property.
    def enterLogger_property(self, ctx:apgParser.Logger_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#logger_property.
    def exitLogger_property(self, ctx:apgParser.Logger_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#log_level.
    def enterLog_level(self, ctx:apgParser.Log_levelContext):
        pass

    # Exit a parse tree produced by apgParser#log_level.
    def exitLog_level(self, ctx:apgParser.Log_levelContext):
        pass


    # Enter a parse tree produced by apgParser#log_format.
    def enterLog_format(self, ctx:apgParser.Log_formatContext):
        pass

    # Exit a parse tree produced by apgParser#log_format.
    def exitLog_format(self, ctx:apgParser.Log_formatContext):
        pass


    # Enter a parse tree produced by apgParser#metric_definition.
    def enterMetric_definition(self, ctx:apgParser.Metric_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#metric_definition.
    def exitMetric_definition(self, ctx:apgParser.Metric_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#metric_type.
    def enterMetric_type(self, ctx:apgParser.Metric_typeContext):
        pass

    # Exit a parse tree produced by apgParser#metric_type.
    def exitMetric_type(self, ctx:apgParser.Metric_typeContext):
        pass


    # Enter a parse tree produced by apgParser#metric_configuration.
    def enterMetric_configuration(self, ctx:apgParser.Metric_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#metric_configuration.
    def exitMetric_configuration(self, ctx:apgParser.Metric_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#metric_property.
    def enterMetric_property(self, ctx:apgParser.Metric_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#metric_property.
    def exitMetric_property(self, ctx:apgParser.Metric_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#unit_specification.
    def enterUnit_specification(self, ctx:apgParser.Unit_specificationContext):
        pass

    # Exit a parse tree produced by apgParser#unit_specification.
    def exitUnit_specification(self, ctx:apgParser.Unit_specificationContext):
        pass


    # Enter a parse tree produced by apgParser#marketplace_entity.
    def enterMarketplace_entity(self, ctx:apgParser.Marketplace_entityContext):
        pass

    # Exit a parse tree produced by apgParser#marketplace_entity.
    def exitMarketplace_entity(self, ctx:apgParser.Marketplace_entityContext):
        pass


    # Enter a parse tree produced by apgParser#marketplace_config.
    def enterMarketplace_config(self, ctx:apgParser.Marketplace_configContext):
        pass

    # Exit a parse tree produced by apgParser#marketplace_config.
    def exitMarketplace_config(self, ctx:apgParser.Marketplace_configContext):
        pass


    # Enter a parse tree produced by apgParser#marketplace_component.
    def enterMarketplace_component(self, ctx:apgParser.Marketplace_componentContext):
        pass

    # Exit a parse tree produced by apgParser#marketplace_component.
    def exitMarketplace_component(self, ctx:apgParser.Marketplace_componentContext):
        pass


    # Enter a parse tree produced by apgParser#user_types_definition.
    def enterUser_types_definition(self, ctx:apgParser.User_types_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#user_types_definition.
    def exitUser_types_definition(self, ctx:apgParser.User_types_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#user_type_list.
    def enterUser_type_list(self, ctx:apgParser.User_type_listContext):
        pass

    # Exit a parse tree produced by apgParser#user_type_list.
    def exitUser_type_list(self, ctx:apgParser.User_type_listContext):
        pass


    # Enter a parse tree produced by apgParser#user_type.
    def enterUser_type(self, ctx:apgParser.User_typeContext):
        pass

    # Exit a parse tree produced by apgParser#user_type.
    def exitUser_type(self, ctx:apgParser.User_typeContext):
        pass


    # Enter a parse tree produced by apgParser#user_type_config.
    def enterUser_type_config(self, ctx:apgParser.User_type_configContext):
        pass

    # Exit a parse tree produced by apgParser#user_type_config.
    def exitUser_type_config(self, ctx:apgParser.User_type_configContext):
        pass


    # Enter a parse tree produced by apgParser#user_type_property.
    def enterUser_type_property(self, ctx:apgParser.User_type_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#user_type_property.
    def exitUser_type_property(self, ctx:apgParser.User_type_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#transaction_engine.
    def enterTransaction_engine(self, ctx:apgParser.Transaction_engineContext):
        pass

    # Exit a parse tree produced by apgParser#transaction_engine.
    def exitTransaction_engine(self, ctx:apgParser.Transaction_engineContext):
        pass


    # Enter a parse tree produced by apgParser#transaction_config.
    def enterTransaction_config(self, ctx:apgParser.Transaction_configContext):
        pass

    # Exit a parse tree produced by apgParser#transaction_config.
    def exitTransaction_config(self, ctx:apgParser.Transaction_configContext):
        pass


    # Enter a parse tree produced by apgParser#transaction_property.
    def enterTransaction_property(self, ctx:apgParser.Transaction_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#transaction_property.
    def exitTransaction_property(self, ctx:apgParser.Transaction_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#trust_safety_system.
    def enterTrust_safety_system(self, ctx:apgParser.Trust_safety_systemContext):
        pass

    # Exit a parse tree produced by apgParser#trust_safety_system.
    def exitTrust_safety_system(self, ctx:apgParser.Trust_safety_systemContext):
        pass


    # Enter a parse tree produced by apgParser#trust_safety_config.
    def enterTrust_safety_config(self, ctx:apgParser.Trust_safety_configContext):
        pass

    # Exit a parse tree produced by apgParser#trust_safety_config.
    def exitTrust_safety_config(self, ctx:apgParser.Trust_safety_configContext):
        pass


    # Enter a parse tree produced by apgParser#trust_safety_property.
    def enterTrust_safety_property(self, ctx:apgParser.Trust_safety_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#trust_safety_property.
    def exitTrust_safety_property(self, ctx:apgParser.Trust_safety_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#search_discovery_engine.
    def enterSearch_discovery_engine(self, ctx:apgParser.Search_discovery_engineContext):
        pass

    # Exit a parse tree produced by apgParser#search_discovery_engine.
    def exitSearch_discovery_engine(self, ctx:apgParser.Search_discovery_engineContext):
        pass


    # Enter a parse tree produced by apgParser#search_config.
    def enterSearch_config(self, ctx:apgParser.Search_configContext):
        pass

    # Exit a parse tree produced by apgParser#search_config.
    def exitSearch_config(self, ctx:apgParser.Search_configContext):
        pass


    # Enter a parse tree produced by apgParser#search_property.
    def enterSearch_property(self, ctx:apgParser.Search_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#search_property.
    def exitSearch_property(self, ctx:apgParser.Search_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#communication_system.
    def enterCommunication_system(self, ctx:apgParser.Communication_systemContext):
        pass

    # Exit a parse tree produced by apgParser#communication_system.
    def exitCommunication_system(self, ctx:apgParser.Communication_systemContext):
        pass


    # Enter a parse tree produced by apgParser#communication_config.
    def enterCommunication_config(self, ctx:apgParser.Communication_configContext):
        pass

    # Exit a parse tree produced by apgParser#communication_config.
    def exitCommunication_config(self, ctx:apgParser.Communication_configContext):
        pass


    # Enter a parse tree produced by apgParser#communication_property.
    def enterCommunication_property(self, ctx:apgParser.Communication_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#communication_property.
    def exitCommunication_property(self, ctx:apgParser.Communication_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#microservices_architecture.
    def enterMicroservices_architecture(self, ctx:apgParser.Microservices_architectureContext):
        pass

    # Exit a parse tree produced by apgParser#microservices_architecture.
    def exitMicroservices_architecture(self, ctx:apgParser.Microservices_architectureContext):
        pass


    # Enter a parse tree produced by apgParser#microservices_config.
    def enterMicroservices_config(self, ctx:apgParser.Microservices_configContext):
        pass

    # Exit a parse tree produced by apgParser#microservices_config.
    def exitMicroservices_config(self, ctx:apgParser.Microservices_configContext):
        pass


    # Enter a parse tree produced by apgParser#microservices_property.
    def enterMicroservices_property(self, ctx:apgParser.Microservices_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#microservices_property.
    def exitMicroservices_property(self, ctx:apgParser.Microservices_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#service_definition.
    def enterService_definition(self, ctx:apgParser.Service_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#service_definition.
    def exitService_definition(self, ctx:apgParser.Service_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#service_config.
    def enterService_config(self, ctx:apgParser.Service_configContext):
        pass

    # Exit a parse tree produced by apgParser#service_config.
    def exitService_config(self, ctx:apgParser.Service_configContext):
        pass


    # Enter a parse tree produced by apgParser#service_property.
    def enterService_property(self, ctx:apgParser.Service_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#service_property.
    def exitService_property(self, ctx:apgParser.Service_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#deployment_configuration.
    def enterDeployment_configuration(self, ctx:apgParser.Deployment_configurationContext):
        pass

    # Exit a parse tree produced by apgParser#deployment_configuration.
    def exitDeployment_configuration(self, ctx:apgParser.Deployment_configurationContext):
        pass


    # Enter a parse tree produced by apgParser#deployment_config.
    def enterDeployment_config(self, ctx:apgParser.Deployment_configContext):
        pass

    # Exit a parse tree produced by apgParser#deployment_config.
    def exitDeployment_config(self, ctx:apgParser.Deployment_configContext):
        pass


    # Enter a parse tree produced by apgParser#deployment_property.
    def enterDeployment_property(self, ctx:apgParser.Deployment_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#deployment_property.
    def exitDeployment_property(self, ctx:apgParser.Deployment_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#internationalization_config.
    def enterInternationalization_config(self, ctx:apgParser.Internationalization_configContext):
        pass

    # Exit a parse tree produced by apgParser#internationalization_config.
    def exitInternationalization_config(self, ctx:apgParser.Internationalization_configContext):
        pass


    # Enter a parse tree produced by apgParser#i18n_config.
    def enterI18n_config(self, ctx:apgParser.I18n_configContext):
        pass

    # Exit a parse tree produced by apgParser#i18n_config.
    def exitI18n_config(self, ctx:apgParser.I18n_configContext):
        pass


    # Enter a parse tree produced by apgParser#i18n_property.
    def enterI18n_property(self, ctx:apgParser.I18n_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#i18n_property.
    def exitI18n_property(self, ctx:apgParser.I18n_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#analytics_intelligence.
    def enterAnalytics_intelligence(self, ctx:apgParser.Analytics_intelligenceContext):
        pass

    # Exit a parse tree produced by apgParser#analytics_intelligence.
    def exitAnalytics_intelligence(self, ctx:apgParser.Analytics_intelligenceContext):
        pass


    # Enter a parse tree produced by apgParser#analytics_config.
    def enterAnalytics_config(self, ctx:apgParser.Analytics_configContext):
        pass

    # Exit a parse tree produced by apgParser#analytics_config.
    def exitAnalytics_config(self, ctx:apgParser.Analytics_configContext):
        pass


    # Enter a parse tree produced by apgParser#analytics_property.
    def enterAnalytics_property(self, ctx:apgParser.Analytics_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#analytics_property.
    def exitAnalytics_property(self, ctx:apgParser.Analytics_propertyContext):
        pass


    # Enter a parse tree produced by apgParser#marketplace_events.
    def enterMarketplace_events(self, ctx:apgParser.Marketplace_eventsContext):
        pass

    # Exit a parse tree produced by apgParser#marketplace_events.
    def exitMarketplace_events(self, ctx:apgParser.Marketplace_eventsContext):
        pass


    # Enter a parse tree produced by apgParser#event_definitions.
    def enterEvent_definitions(self, ctx:apgParser.Event_definitionsContext):
        pass

    # Exit a parse tree produced by apgParser#event_definitions.
    def exitEvent_definitions(self, ctx:apgParser.Event_definitionsContext):
        pass


    # Enter a parse tree produced by apgParser#event_definition.
    def enterEvent_definition(self, ctx:apgParser.Event_definitionContext):
        pass

    # Exit a parse tree produced by apgParser#event_definition.
    def exitEvent_definition(self, ctx:apgParser.Event_definitionContext):
        pass


    # Enter a parse tree produced by apgParser#event_config.
    def enterEvent_config(self, ctx:apgParser.Event_configContext):
        pass

    # Exit a parse tree produced by apgParser#event_config.
    def exitEvent_config(self, ctx:apgParser.Event_configContext):
        pass


    # Enter a parse tree produced by apgParser#event_property.
    def enterEvent_property(self, ctx:apgParser.Event_propertyContext):
        pass

    # Exit a parse tree produced by apgParser#event_property.
    def exitEvent_property(self, ctx:apgParser.Event_propertyContext):
        pass



del apgParser