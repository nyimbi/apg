# Claude Custom Commands for APG Development

This directory contains custom Claude commands to automate and enhance the development process for the APG (Application Programming Generation) ERP system.

## Available Commands

### `/dev` - Advanced Capability Development

The `/dev` command automates the complete development lifecycle for ERP capabilities and sub-capabilities with industry-leading features.

#### Usage
```
/dev <capability_name>/<sub_capability_name>
```

#### Examples
```
/dev general_cross_functional/customer_relationship_management
/dev core_financials/accounts_payable
/dev human_resources/employee_management
/dev manufacturing/production_planning
```

#### What it does

1. **Analyzes** the requested capability and researches industry best practices
2. **Creates** a comprehensive capability specification (`cap_spec.md`)
3. **Generates** a detailed development plan (`todo.md`)
4. **Implements** complete data models with advanced features
5. **Builds** rich business logic with AI integration and background processing
6. **Creates** modern, responsive UI views with accessibility compliance
7. **Develops** comprehensive REST APIs with documentation
8. **Writes** extensive test suites (unit, integration, performance, security)
9. **Generates** complete documentation in `docs/` directory (user, developer guides)
10. **Identifies** 10 world-class functionality improvements for competitive advantage
11. **Implements** the 10 world-class  functionality improvements carefully and meticulously
12. **Ensures** enterprise-grade quality and compliance

#### Features Implemented

- **AI/ML Integration**: Predictive analytics, intelligent automation, NLP
- **Modern UI/UX**: Mobile-first design, real-time updates, accessibility
- **Enterprise Features**: Multi-tenancy, RBAC, audit trails, SSO
- **Performance**: Horizontal scaling, caching, optimization
- **Security**: OWASP compliance, encryption, vulnerability scanning
- **Integration**: REST/GraphQL APIs, webhooks, event-driven architecture
- **Background Processing**: Celery integration, workflow automation
- **Monitoring**: Observability, analytics, performance tracking

## File Structure Created

When you run `/dev capability/sub_capability`, it creates:

```
capabilities/capability/sub_capability/
├── cap_spec.md              # Comprehensive specification
├── todo.md                  # Detailed development plan
├── __init__.py              # Metadata and configuration
├── models.py                # Advanced data models
├── service.py               # Business logic + background processing
├── views.py                 # Rich UI with modern features
├── api.py                   # Comprehensive REST API
├── blueprint.py             # Flask integration
├── WORLD_CLASS_IMPROVEMENTS.md # 10 revolutionary enhancements
├── docs/                    # Complete documentation
│   ├── user_guide.md
│   ├── developer_guide.md
│   ├── api_reference.md
│   ├── installation_guide.md
│   └── troubleshooting_guide.md
├── tests/                   # Complete test suite
│   ├── test_models.py
│   ├── test_service.py
│   ├── test_api.py
│   ├── test_views.py
│   ├── test_performance.py
│   ├── test_security.py
│   ├── test_integration.py
│   ├── fixtures/
│   ├── test_data/
│   └── conftest.py
├── static/                  # Frontend assets
│   ├── css/
│   ├── js/
│   └── images/
└── templates/               # Jinja2 templates
    ├── base/
    ├── forms/
    └── dashboards/
```

## Quality Standards

Every capability developed with `/dev` meets enterprise standards:

- **Code Quality**: SOLID principles, clean architecture, type hints
- **Testing**: >95% coverage, unit/integration/performance/security tests
- **Security**: OWASP Top 10 compliance, vulnerability scanning
- **Performance**: Sub-second response times, horizontal scaling
- **Accessibility**: WCAG 2.1 AA compliance
- **Documentation**: Complete user/developer/integration guides
- **AI Integration**: Modern ML/AI features where applicable

## Templates

The command uses sophisticated templates in `templates/` directory:

- `cap_spec_template.md` - Capability specification template
- `todo_template.md` - Development plan template
- `test_template.py` - Comprehensive testing template

## Utilities

Helper functions in `helpers/dev_utils.py` provide:

- Capability path parsing and validation
- Directory structure creation
- Template processing and variable substitution
- Industry best practice recommendations
- Development time estimation
- Project planning and milestone tracking

## Getting Started

1. Ensure you have the `.claude/commands/` directory in your project
2. Use `/dev capability/sub_capability` to start development
3. Follow the generated `todo.md` plan
4. Review the `cap_spec.md` for requirements
5. Run tests as you implement features
6. Use the generated documentation for deployment

## Example Usage

To develop a new Customer Relationship Management sub-capability:

```
/dev general_cross_functional/customer_relationship_management
```

This will:
1. Analyze CRM industry standards and best practices
2. Create a comprehensive specification with AI-powered features
3. Generate a detailed 18-day development plan
4. Implement 25+ database models with relationships
5. Build rich business logic with automation
6. Create responsive UI with real-time features
7. Develop comprehensive APIs with documentation
8. Write extensive test coverage
9. Generate complete user and developer guides

The result is a production-ready, enterprise-grade CRM system with modern features like:
- AI-powered lead scoring and recommendations
- Real-time collaboration and notifications
- Advanced analytics and forecasting
- Mobile-optimized responsive design
- Complete accessibility compliance
- Comprehensive security and audit trails
- Integration-ready APIs and webhooks
- 10 revolutionary improvements that surpass world-class competitors
- implemented the 10 world-class functionality improvements carefully and meticulously

## Benefits

- **Speed**: Automated development reduces time by 80%
- **Quality**: Consistent enterprise-grade standards
- **Completeness**: Nothing is missed - complete implementation
- **Modern**: Latest technologies and best practices
- **Maintainable**: Clean, documented, tested code
- **Scalable**: Enterprise architecture patterns
- **Compliant**: Security, accessibility, and regulatory compliance
- **Extensible**: Easy to customize and extend

Use `/dev` to rapidly build world-class ERP capabilities with modern, AI-powered features!
