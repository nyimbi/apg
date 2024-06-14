grammar AppGen;

// Parser Rules
projectBlock: projectStructure cloudDeployment databaseAndCaching authenticationAndAuthorization thirdPartyIntegration customization performanceOptimization versionManagementAndMaintenance pluginAndExtensionSystem EOF;

projectStructure: PROJECT_STRUCTURE STRING_VALUE;
cloudDeployment: CLOUD_DEPLOYMENT cloudPlatform;
cloudPlatform: AWS | AZURE | LINODE;
databaseAndCaching: DATABASE_AND_CACHING databaseType cachingType;
databaseType: MYSQL | POSTGRESQL;
cachingType: REDIS | MEMCACHED;

authenticationAndAuthorization: AUTHENTICATION_AND_AUTHORIZATION authScheme;
authScheme: OAUTH | SAML | OPENID | JWT | LDAP | FLASK_APPBUILDER_AUTH;

thirdPartyIntegration: THIRD_PARTY_INTEGRATION integrationDetails;
integrationDetails: SINGER_IO_TAPS (STRING_VALUE (COMMA STRING_VALUE)*);

customization: CUSTOMIZATION customizationOptions;
customizationOptions: WIDGET_PLACEMENT | THEMING | LOOK_AND_FEEL;

performanceOptimization: PERFORMANCE_OPTIMIZATION optimizationType;
optimizationType: LOAD_BALANCING | CACHING_STRATEGY | RESOURCE_OPTIMIZATION;

versionManagementAndMaintenance: VERSION_MANAGEMENT_AND_MAINTENANCE maintenanceOptions;
maintenanceOptions: AUTO_UPDATES | BUG_FIXES;

pluginAndExtensionSystem: PLUGIN_AND_EXTENSION_SYSTEM pluginOrExtensionDetails;
pluginOrExtensionDetails: PLUGIN_OR_EXTENSION STRING_VALUE;

// Lexer Rules
PROJECT_STRUCTURE: 'project_structure';
CLOUD_DEPLOYMENT: 'cloud_deployment';
DATABASE_AND_CACHING: 'database_and_caching';
AUTHENTICATION_AND_AUTHORIZATION: 'authentication_and_authorization';
THIRD_PARTY_INTEGRATION: 'third_party_integration';
CUSTOMIZATION: 'customization';
PERFORMANCE_OPTIMIZATION: 'performance_optimization';
VERSION_MANAGEMENT_AND_MAINTENANCE: 'version_management_and_maintenance';
PLUGIN_AND_EXTENSION_SYSTEM: 'plugin_and_extension_system';

AWS: 'aws';
AZURE: 'azure';
LINODE: 'linode';

MYSQL: 'mysql';
POSTGRESQL: 'postgresql';

REDIS: 'redis';
MEMCACHED: 'memcached';

OAUTH: 'oauth';
SAML: 'saml';
OPENID: 'openid';
JWT: 'jwt';
LDAP: 'ldap';
FLASK_APPBUILDER_AUTH: 'flask_appbuilder_auth';

SINGER_IO_TAPS: 'singer_io_taps';

WIDGET_PLACEMENT: 'widget_placement';
THEMING: 'theming';
LOOK_AND_FEEL: 'look_and_feel';

LOAD_BALANCING: 'load_balancing';
CACHING_STRATEGY: 'caching_strategy';
RESOURCE_OPTIMIZATION: 'resource_optimization';

AUTO_UPDATES: 'auto_updates';
BUG_FIXES: 'bug_fixes';

PLUGIN_OR_EXTENSION: 'plugin_or_extension';

COMMA: ',';
STRING_VALUE: '"' ('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'-'|'.'|'/')* '"';
WS: [ \t\r\n]+ -> skip;
