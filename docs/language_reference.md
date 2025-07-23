# APG Language Reference Guide

**APG v11 - Ultra-Terse Application Programming Generation Language**

Copyright (c) 2025 Datacraft  
Author: Nyimbi Odero

---

## Table of Contents

1. [Language Overview](#language-overview)
2. [Universal Entity Pattern](#universal-entity-pattern)
3. [Core Language Features](#core-language-features)
4. [Data Types and Type System](#data-types-and-type-system)
5. [Entity Types](#entity-types)
6. [Configuration Syntax](#configuration-syntax)
7. [Behavior Definitions](#behavior-definitions)
8. [Control Flow](#control-flow)
9. [Expressions and Operators](#expressions-and-operators)
10. [Advanced Features](#advanced-features)
11. [Domain-Specific Extensions](#domain-specific-extensions)
12. [Compilation and Runtime](#compilation-and-runtime)

---

## Language Overview

APG is a domain-specific language designed for ultra-terse, highly productive application development across multiple domains. The language compiles to high-quality Python code and supports everything from simple web applications to complex digital twins and AI systems.

### Design Principles

- **Ultra-Terse**: 10-30 lines instead of thousands
- **Universal Pattern**: One syntax pattern fits all domains
- **Consistent**: Same patterns across different application types
- **Productive**: Minimize cognitive load and maximize expressiveness
- **Modern**: Leverages latest Python features and best practices

### Key Characteristics

- **Declarative Syntax**: Focus on what, not how
- **Type-Safe**: Full type system with inference
- **Async-First**: Native support for concurrent operations
- **Domain-Agnostic**: Single language for multiple problem domains
- **Python-Compilable**: Generates clean, readable Python code

---

## Universal Entity Pattern

The core of APG is the universal entity pattern that scales from simple data structures to complex systems:

```apg
type name {
    config: value;
    behavior: action;
}
```

### Pattern Components

1. **Type**: Specifies the entity category and capabilities
2. **Name**: Unique identifier for the entity instance
3. **Config**: Declarative configuration properties
4. **Behavior**: Executable actions and workflows

### Pattern Variations

```apg
// Minimal form
db UserDB { users: table { name: str; }; }

// With decorators
@secure @cached
agent Assistant { models: gpt4->claude3; }

// With inheritance
robot MobileRobot extends BaseRobot { 
    locomotion: differential_drive; 
}

// With version control
twin Factory version 2.1.0 {
    sync: real_time_sync("opc.tcp://192.168.1.100:4840");
}
```

---

## Core Language Features

### Modern Python Integration

APG generates Python 3.12+ code with full type annotations:

```apg
// APG Code
calc TaxCalculation {
    income: decimal;
    rate: decimal [range=0.0..1.0];
    deduction: decimal?;
    
    tax_owed: decimal = (income - (deduction ?? 0)) * rate;
}

// Generated Python
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field

class TaxCalculation(BaseModel):
    income: Decimal
    rate: Decimal = Field(ge=0.0, le=1.0)
    deduction: Optional[Decimal] = None
    
    @property
    def tax_owed(self) -> Decimal:
        return (self.income - (self.deduction or Decimal('0'))) * self.rate
```

### Async/Await Support

Native asynchronous programming support:

```apg
agent AsyncProcessor {
    async def process_stream() -> list[str] {
        data = await sensor.read();
        results = await ml_model.predict(data);
        await database.save(results);
        return results;
    }
}
```

### Pattern Matching

Advanced control flow with pattern matching:

```apg
match sensor_reading {
    case Temperature(value) if value > 80 -> alert("Overheating");
    case Pressure(value) if value < 10 -> alert("Low pressure");
    case Normal() -> continue_operation();
    case _ -> investigate_anomaly();
}
```

---

## Data Types and Type System

### Basic Types

```apg
// Primitive types
name: str;
age: int;
price: float;
active: bool;
data: bytes;
timestamp: datetime;
amount: decimal;

// Nullable types
optional_field: str?;
maybe_number: int | None;
```

### Collection Types

```apg
// Lists
tags: list[str];
numbers: [int];  // Shorthand syntax
coordinates: list[float] = [1.0, 2.0, 3.0];

// Dictionaries
metadata: dict[str, str];
config: {str: Any};  // Shorthand syntax
settings: dict[str, int] = {"timeout": 30, "retries": 3};
```

### Union Types

```apg
// Union types (Python 3.10+ style)
result: str | int | None;
status: "pending" | "processing" | "completed";
response: Success | Error | Timeout;
```

### Generic Types

```apg
// Generic containers
queue: Queue[Task];
cache: Cache[str, UserData];
repository: Repository[User, int];
```

### Custom Types and Constraints

```apg
// Constrained types
email: str [email, required];
password: str [min_length=8, pattern=/^(?=.*[A-Za-z])(?=.*\d)/];
score: int [range=0..100];
rate: float [min_value=0.0, max_value=1.0];
```

---

## Entity Types

APG supports a comprehensive set of entity types across multiple domains:

### Core Application Types

```apg
// Database entities
db UserDatabase { users: table { id: int [pk]; name: str; }; }
table Products { name: str; price: decimal; stock: int; }

// User interface entities
form ContactForm { fields: name, email, message; }
report SalesReport { source: sales_data; fields: date, amount, customer; }
dashboard ExecutiveDashboard { widgets: revenue_chart, sales_metrics; }

// Business logic entities
biz OrderProcessing { 
    workflow: validate -> process_payment -> fulfill -> ship;
}
rule DiscountRule { 
    when: order_total > 100 -> apply_discount(10%);
}
```

### Digital Twin and Industrial Types

```apg
// Digital twin entities
twin ManufacturingLine {
    physical_model: "Assembly_Line_A1";
    sync: real_time_sync("opc.tcp://192.168.1.100:4840");
}

// Simulation entities
simulate FluidDynamics {
    type: computational_fluid_dynamics;
    mesh_resolution: 0.5mm;
    solver: "ansys_fluent";
}

// Industrial monitoring
monitor ProductionMonitor {
    sensors: [temperature, pressure, flow_rate];
    thresholds: {temperature: 85°C, pressure: 150psi};
}
```

### AI and Robotics Types

```apg
// AI agent entities
agent CustomerService {
    models: gpt4->claude3->llama2;
    memory: conversational + factual;
}

// Robotics entities
robot ManipulatorArm {
    dof: 6;
    payload: 10kg;
    reach: 1.2m;
    control: position + velocity + force;
}

// Sensor entities
sensor VisionSensor {
    type: "basler_camera";
    resolution: "2048x1536";
    frame_rate: 30fps;
}
```

### Intelligence and Analytics Types

```apg
// OSINT entities
intel ThreatIntelligence {
    sources: [social_media, dark_web, government_feeds];
    analysis: [correlation, attribution, prediction];
}

// Analysis entities
analyze DataAnalyzer {
    algorithms: [clustering, classification, regression];
    pipelines: [preprocessing, feature_extraction, modeling];
}

// Visualization entities
graph NetworkGraph {
    nodes: entities;
    edges: relationships;
    layout: force_directed;
}
```

---

## Configuration Syntax

### Basic Configuration

```apg
entity Example {
    // Simple key-value pairs
    name: "Production Server";
    port: 8080;
    enabled: true;
    
    // Complex values
    database_url: "postgresql://user:pass@localhost:5432/mydb";
    tags: ["production", "web-server", "critical"];
    
    // Environment variables
    secret_key: $SECRET_KEY;
    debug_mode: env("DEBUG_MODE");
}
```

### Nested Configuration

```apg
server WebServer {
    config: {
        host: "0.0.0.0";
        port: 8080;
        ssl: {
            enabled: true;
            cert_file: "/etc/ssl/server.crt";
            key_file: "/etc/ssl/server.key";
        };
        database: {
            url: "postgresql://localhost:5432/mydb";
            pool_size: 20;
            timeout: 30s;
        };
    };
}
```

### Cascade Configuration

```apg
// Fallback chains for resilience
chat Assistant {
    models: gpt4->claude3->llama2->local_model;
    databases: primary_db->replica_db->cache->static_data;
    servers: main_server->backup_server->emergency_fallback;
}
```

### Reference Configuration

```apg
// Object references
workflow OrderProcessing {
    database: UserDB.orders;  // Reference to table
    payment_service: PaymentGateway;  // Reference to service
    notification: email_service + sms_service;  // Combination
}
```

---

## Behavior Definitions

### Annotations

Annotations provide metadata and configuration for behaviors:

```apg
agent ProcessingAgent {
    @async: parallel_processing + batch_size(100);
    @retry: max_attempts(3) + backoff(exponential);
    @cache: ttl(300s) + eviction_policy(lru);
    @security: authentication_required + rate_limit(100/minute);
    
    process_data(input: list[str]) -> list[Result];
}
```

### Method Definitions

```apg
// Simple method
def calculate_tax(income: decimal, rate: decimal) -> decimal {
    return income * rate;
}

// Async method
async def fetch_data(url: str) -> dict[str, Any] {
    response = await http_client.get(url);
    return await response.json();
}

// Lambda expressions
transform: data => data.filter(valid=true).map(normalize);
validator: value => value > 0 && value < 100;
```

### Flow Definitions

```apg
// Sequential flow
flow DataPipeline {
    extract_data -> clean_data -> transform_data -> load_data -> validate_results;
}

// Parallel flow
flow ParallelProcessing {
    input_data -> [process_batch_1 || process_batch_2 || process_batch_3] -> merge_results;
}

// Conditional flow
flow ConditionalWorkflow {
    validate_input -> 
    if (valid) { process_data -> save_results; }
    else { log_error -> send_notification; };
}
```

### Event Handling

```apg
// Event-driven behaviors
when: temperature > 80°C -> {
    alert.send("High temperature detected");
    cooling_system.activate();
    log.warning("Temperature exceeded threshold");
};

// Multiple conditions
when: {
    pressure < 10psi -> emergency_shutdown();
    vibration > 5mm/s -> schedule_maintenance();
    efficiency < 80% -> optimize_parameters();
};
```

---

## Control Flow

### Conditional Statements

```apg
// If-else statements
if (temperature > threshold) {
    activate_cooling();
} elif (temperature < min_threshold) {
    activate_heating();
} else {
    maintain_temperature();
}

// Ternary operator
status = temperature > 80 ? "hot" : "normal";
```

### Loops

```apg
// For loops
for sensor in sensors {
    reading = sensor.read();
    process_reading(reading);
}

// While loops
while (system.is_running()) {
    data = collect_data();
    process_data(data);
    await sleep(1s);
}

// Async loops
async for item in async_generator() {
    result = await process_item(item);
    await save_result(result);
}
```

### Pattern Matching

```apg
// Basic pattern matching
match response {
    case Success(data) -> process_data(data);
    case Error(code, message) -> handle_error(code, message);
    case Timeout() -> retry_request();
    case _ -> log_unknown_response();
}

// With guards
match sensor_data {
    case Temperature(value) if value > 100 -> emergency_shutdown();
    case Temperature(value) if value > 80 -> activate_cooling();
    case Temperature(value) -> normal_operation();
    case _ -> sensor_error();
}
```

### Exception Handling

```apg
// Try-catch blocks
try {
    result = process_data(input);
    save_result(result);
} except ValueError as e {
    log.error(f"Invalid data: {e}");
    return default_result();
} except ConnectionError {
    log.error("Database connection failed");
    retry_later();
} finally {
    cleanup_resources();
}
```

---

## Expressions and Operators

### Arithmetic Operators

```apg
// Basic arithmetic
total = price * quantity + tax;
average = sum(values) / count(values);
compound_interest = principal * (1 + rate) ** years;

// Operator precedence (follows Python)
result = a + b * c ** d;  // Same as: a + (b * (c ** d))
```

### Comparison Operators

```apg
// Comparison operations
is_valid = score >= 0 && score <= 100;
is_member = name in allowed_users;
is_instance = obj is User;
not_equal = value != expected;
```

### Logical Operators

```apg
// Boolean logic
can_proceed = user.is_authenticated() && user.has_permission("read");
should_retry = attempt_count < max_attempts || is_critical_operation;
is_invalid = !validator.check(data);
```

### Pipeline Operators

```apg
// Data transformation pipelines
result = data 
    | filter(lambda x: x.is_valid()) 
    | map(lambda x: x.transform()) 
    | reduce(lambda acc, x: acc + x.value, 0);

// Method chaining alternative
result = data.filter(valid=true)
             .map(transform)
             .reduce(sum_values);
```

### String Operations

```apg
// F-strings and formatting
message = f"Hello {user.name}, your balance is ${account.balance:.2f}";
template = "User: {name}, Age: {age}";
formatted = template.format(name="John", age=30);

// String methods
clean_text = input_text.strip().lower().replace(" ", "_");
```

---

## Advanced Features

### Decorators

```apg
// Function decorators
@cache(ttl=300)
@retry(max_attempts=3)
@measure_performance
def expensive_computation(data: list[float]) -> float {
    return complex_calculation(data);
}

// Class decorators
@dataclass
@validate_on_assignment
entity User {
    name: str;
    email: str [email];
    age: int [range=0..150];
}
```

### Generics and Type Variables

```apg
// Generic functions
def process_items<T>(items: list[T], processor: Callable[[T], T]) -> list[T] {
    return [processor(item) for item in items];
}

// Generic classes
entity Repository<T, K> {
    storage: dict[K, T];
    
    def get(key: K) -> T? {
        return storage.get(key);
    }
    
    def save(key: K, item: T) -> None {
        storage[key] = item;
    }
}
```

### Context Managers

```apg
// With statements
with database.transaction() as tx {
    user = tx.query(User).filter(id=user_id).first();
    user.balance += amount;
    tx.save(user);
}

// Async context managers
async with api_client.session() as session {
    response = await session.get("/api/data");
    data = await response.json();
}
```

### Comprehensions

```apg
// List comprehensions
valid_scores = [score for score in scores if score >= 0];
squared = [x**2 for x in range(10)];

// Dict comprehensions
user_lookup = {user.id: user.name for user in users};
config_dict = {key: env(key) for key in config_keys};

// Set comprehensions
unique_domains = {email.split("@")[1] for email in email_list};
```

---

## Domain-Specific Extensions

### Marketplace and E-commerce Extensions

APG v11 includes comprehensive support for building two-sided and multi-sided marketplaces, matching platforms, and e-commerce solutions.

```apg
// Two-sided marketplace definition
marketplace FreelanceHub {
    user_types: {
        client: {
            permissions: ["create_projects", "browse_freelancers", "make_payments"];
            verification_required: {
                email: true;
                payment_method: true;
            };
            commission_structure: {
                platform_fee: 0%;
                payment_processing_fee: 2.9%;
            };
        };
        
        freelancer: {
            permissions: ["create_profile", "submit_proposals", "receive_payments"];
            verification_required: {
                identity: true;
                professional_credentials: true;
            };
            commission_structure: {
                platform_fee: 8%;
                payment_processing_fee: 2.9%;
            };
        };
    };
    
    transactions: {
        escrow_enabled: true;
        payment_providers: ["stripe", "paypal", "cryptocurrency"];
        supported_currencies: ["USD", "EUR", "GBP"];
        dispute_resolution: {
            auto_resolution_enabled: true;
            mediation_process: {
                stages: ["automated_review", "peer_mediation", "admin_arbitration"];
            };
        };
    };
    
    trust_safety: {
        identity_verification: {
            levels: {
                basic: {requirements: ["email", "phone"]};
                premium: {requirements: ["government_id", "address_proof"]};
            };
        };
        
        rating_system: {
            scale: 1..5;
            categories: ["communication", "quality_of_work", "adherence_to_deadline"];
        };
        
        fraud_prevention: {
            ml_models: ["payment_fraud_detector", "account_takeover_detector"];
            behavioral_analysis: true;
        };
    };
    
    search_discovery: {
        search_engine: {
            provider: "elasticsearch";
            features: ["fuzzy_matching", "faceted_search", "geolocation"];
        };
        
        recommendation_engine: {
            algorithms: {
                collaborative_filtering: {weight: 0.4};
                content_based: {weight: 0.3};
                popularity: {weight: 0.3};
            };
        };
    };
}
```

### Microservices Architecture

APG provides native support for defining and deploying microservices architectures:

```apg
// Service definition with placement strategy
service user_service {
    type: "core_domain_service";
    responsibilities: [
        "user_registration", "profile_management", 
        "authentication", "authorization"
    ];
    
    api_endpoints: [
        {
            path: "/api/v1/users";
            methods: ["GET", "POST", "PUT", "DELETE"];
            authentication: "jwt_required";
            rate_limit: "100/minute/user";
        }
    ];
    
    database: {
        type: "postgresql";
        name: "users_db";
        tables: ["users", "profiles", "verifications"];
        partitioning: {
            strategy: "range";
            column: "created_at";
            interval: "monthly";
        };
    };
    
    scaling: {
        strategy: "horizontal";
        min_instances: 3;
        max_instances: 20;
        cpu_threshold: 70%;
        custom_metrics: [
            {
                name: "active_sessions";
                threshold: 1000;
                scale_up: 2;
            }
        ];
    };
    
    deployment: {
        container: {
            image: "company/user-service:v2.3.1";
            resources: {
                requests: {cpu: "200m", memory: "512Mi"};
                limits: {cpu: "1000m", memory: "2Gi"};
            };
        };
        
        placement_strategy: {
            type: "multi_az_spread";
            constraints: ["node.role == worker"];
            preferences: ["zone == us-west-2a"];
        };
        
        networking: {
            service_mesh: "istio";
            protocols: ["grpc", "http"];
            security: "mutual_tls";
        };
    };
    
    monitoring: {
        metrics: {
            business: ["user_registrations_per_minute", "active_sessions"];
            technical: ["request_duration", "error_rate"];
        };
        
        alerts: [
            {
                name: "high_error_rate";
                condition: "error_rate > 5%";
                severity: "critical";
                notification: ["pagerduty", "slack"];
            }
        ];
    };
}

// Service mesh configuration
microservices: {
    api_gateway: {
        provider: "kong";
        features: {
            rate_limiting: true;
            authentication: ["jwt", "oauth2"];
            load_balancing: "round_robin";
        };
    };
    
    service_mesh: {
        provider: "istio";
        features: {
            traffic_management: true;
            security_policies: true;
            mutual_tls: "strict";
        };
    };
    
    service_discovery: {
        provider: "consul";
        health_checking: true;
        automatic_registration: true;
    };
}
```

### Advanced Deployment Strategies

```apg
// Multi-region deployment strategy
deployment_strategy MultiRegionDeployment {
    regions: {
        primary: {
            name: "us-west-2";
            services: ["all"];
            capacity: "100%";
        };
        
        secondary: {
            name: "us-east-1";
            services: ["user_service", "payment_service"];
            capacity: "75%";
        };
    };
    
    service_placement_matrix: {
        user_service: {
            placement: "all_regions";
            replication_strategy: "active_active";
            data_consistency: "eventual";
        };
        
        payment_service: {
            placement: "primary_and_secondary";
            replication_strategy: "active_passive";
            data_consistency: "strong";
            compliance_requirements: ["pci_dss"];
        };
    };
    
    disaster_recovery: {
        rto: "5_minutes";
        rpo: "30_seconds";
        automatic_failover: true;
    };
}

// Canary deployment pattern
deployment_pattern CanaryDeployment {
    stages: [
        {
            name: "initial_canary";
            traffic_percentage: 5%;
            duration: "10m";
            success_criteria: {
                error_rate: "<1%";
                latency_p99: "<500ms";
            };
        },
        {
            name: "full_rollout";
            traffic_percentage: 100%;
            duration: "monitored";
        }
    ];
    
    rollback_triggers: [
        {
            condition: "error_rate > 2%";
            action: "immediate_rollback";
        }
    ];
}
```

### Digital Twin Extensions

```apg
// Physics simulation
@physics: finite_element {
    mesh_resolution: 0.1mm;
    material_properties: steel_316;
    boundary_conditions: fixed_base + rotating_spindle;
    analysis_type: static + dynamic + thermal;
}

// Real-time synchronization
@sync: real_time_sync {
    protocol: opc_ua;
    frequency: 1000Hz;
    security: sign_and_encrypt;
    failover: backup_plc;
}
```

### Computer Vision Extensions

```apg
// Image processing pipeline
@vision_pipeline: {
    preprocessing: gaussian_blur(σ=1.0) + histogram_equalization;
    feature_extraction: sift + surf + orb;
    detection: yolo_v8 + confidence_threshold(0.8);
    tracking: kalman_filter + hungarian_algorithm;
}

// Quality control metrics
@quality_control: {
    dimensional_accuracy: ±0.01mm;
    surface_roughness: Ra<1.6μm;
    color_consistency: ΔE<2.0;
    defect_classification: cnn_model("defect_classifier_v2.1");
}
```

### AI Agent Extensions

```apg
// Cognitive architecture
@cognitive: {
    perception: multimodal_fusion + attention_mechanism;
    memory: episodic + semantic + working + procedural;
    reasoning: chain_of_thought + tree_of_thoughts + logical_inference;
    planning: hierarchical + temporal + resource_aware;
    learning: few_shot + transfer + continual + meta_learning;
}

// Multi-agent coordination
@coordination: {
    communication: message_passing + shared_memory + blackboard;
    negotiation: auction_based + contract_net + consensus_building;
    collaboration: task_allocation + resource_sharing + joint_planning;
    competition: game_theory + nash_equilibrium + mechanism_design;
}
```

### Business Process Extensions

```apg
// Workflow modeling
@workflow: {
    process_model: bpmn_2_0;
    execution_engine: zeebe + camunda;
    monitoring: process_mining + performance_analytics;
    optimization: bottleneck_analysis + resource_allocation;
}

// Compliance frameworks
@compliance: {
    regulations: sox + gdpr + hipaa + pci_dss;
    audit_trail: immutable_log + digital_signatures + timestamps;
    controls: segregation_of_duties + approval_workflows + risk_assessment;
    reporting: regulatory_reports + management_dashboards + exception_alerts;
}
```

---

## Compilation and Runtime

### Compilation Process

1. **Lexical Analysis**: Tokenize APG source code
2. **Parsing**: Build Abstract Syntax Tree (AST) using ANTLR grammar
3. **Semantic Analysis**: Type checking, validation, and optimization
4. **Code Generation**: Generate clean Python code with type hints
5. **Runtime Integration**: Bind to frameworks and libraries

### Generated Code Characteristics

```apg
// APG Input
db UserDB {
    users: table {
        id: int [pk, increment];
        name: str [required, max_length=100];
        email: str [email, unique];
        created_at: datetime [default=now()];
    };
}

// Generated Python Output
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())

class UserSchema(BaseModel):
    id: Optional[int] = None
    name: str = Field(..., max_length=100)
    email: EmailStr
    created_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
        validate_assignment = True
```

### Runtime Libraries

APG generates code that integrates with:

- **Web Frameworks**: FastAPI, Django, Flask
- **Database ORMs**: SQLAlchemy, Django ORM, Tortoise ORM
- **ML Frameworks**: PyTorch, TensorFlow, scikit-learn
- **Industrial Protocols**: python-opcua, pymodbus, snap7
- **Computer Vision**: OpenCV, PIL, scikit-image
- **AI Libraries**: LangChain, transformers, openai

### Performance Optimizations

- **Async by Default**: All I/O operations are asynchronous
- **Lazy Loading**: Database relationships loaded on demand
- **Caching**: Intelligent caching of computation results
- **Type Checking**: Runtime type validation when needed
- **Resource Management**: Automatic connection pooling and cleanup

---

## Best Practices

### Code Organization

```apg
// Organize by domain
// File: user_management.apg
db UserDB { ... }
form UserForm { ... }
api UserAPI { ... }

// File: order_processing.apg  
biz OrderProcessing { ... }
workflow OrderWorkflow { ... }
rule BusinessRules { ... }
```

### Error Handling

```apg
// Comprehensive error handling
try {
    result = risky_operation();
} except SpecificError as e {
    log.error(f"Specific error: {e}");
    return fallback_result();
} except Exception as e {
    log.error(f"Unexpected error: {e}");
    notify_administrators(e);
    raise;
}
```

### Testing Integration

```apg
// Built-in testing support
@test: {
    unit_tests: pytest + coverage(90%);
    integration_tests: docker_compose + test_database;
    load_tests: locust + performance_benchmarks;
    security_tests: bandit + safety + semgrep;
}
```

### Documentation

```apg
// Self-documenting code
entity DocumentedExample {
    /// This field stores user preferences
    /// @example: {"theme": "dark", "language": "en"}
    preferences: dict[str, str];
    
    /// Calculate user score based on activity
    /// @param activity_data: User activity metrics
    /// @returns: Score between 0 and 100
    def calculate_score(activity_data: ActivityData) -> int;
}
```

---

This language reference provides a comprehensive guide to APG syntax, features, and capabilities. For additional examples and tutorials, see the accompanying documentation and example files.