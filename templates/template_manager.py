#!/usr/bin/env python3
"""
APG Template Manager
===================

Manages APG project templates and provides template generation functionality.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .template_types import TemplateType, ProjectConfig, get_template_info


class TemplateManager:
	"""Manages APG project templates"""
	
	def __init__(self, templates_directory: Optional[Path] = None):
		"""Initialize template manager"""
		if templates_directory is None:
			templates_directory = Path(__file__).parent / "templates"
		
		self.templates_dir = Path(templates_directory)
		self.templates_dir.mkdir(exist_ok=True)
		
		# Initialize built-in templates if they don't exist
		self._ensure_builtin_templates()
	
	def _ensure_builtin_templates(self):
		"""Ensure built-in templates exist"""
		for template_type in TemplateType:
			template_dir = self.templates_dir / template_type.value
			if not template_dir.exists():
				self._create_builtin_template(template_type)
	
	def _create_builtin_template(self, template_type: TemplateType):
		"""Create a built-in template"""
		template_dir = self.templates_dir / template_type.value
		template_dir.mkdir(parents=True, exist_ok=True)
		
		# Create template configuration
		config = self._get_template_config(template_type)
		config_file = template_dir / "template.json"
		
		with open(config_file, 'w') as f:
			json.dump(config, f, indent=2)
		
		# Create APG source template
		apg_content = self._get_template_apg_content(template_type)
		apg_file = template_dir / "main.apg.template"
		
		with open(apg_file, 'w') as f:
			f.write(apg_content)
		
		# Create additional template files
		additional_files = self._get_template_additional_files(template_type)
		for filename, content in additional_files.items():
			file_path = template_dir / filename
			file_path.parent.mkdir(parents=True, exist_ok=True)
			
			with open(file_path, 'w') as f:
				f.write(content)
	
	def _get_template_config(self, template_type: TemplateType) -> Dict[str, Any]:
		"""Get template configuration"""
		info = get_template_info(template_type)
		
		base_config = {
			'name': info.get('name', template_type.value.replace('_', ' ').title()),
			'description': info.get('description', f'{template_type.value} template'),
			'complexity': info.get('complexity', 'Intermediate'),
			'features': info.get('features', []),
			'use_cases': info.get('use_cases', []),
			'variables': self._get_template_variables(template_type),
			'files': self._get_template_file_list(template_type)
		}
		
		return base_config
	
	def _get_template_variables(self, template_type: TemplateType) -> Dict[str, Any]:
		"""Get template variables for substitution"""
		base_vars = {
			'project_name': '{{project_name}}',
			'project_description': '{{project_description}}',
			'author': '{{author}}',
			'version': '{{version}}',
			'license': '{{license}}'
		}
		
		# Add template-specific variables
		if template_type == TemplateType.TASK_MANAGEMENT:
			base_vars.update({
				'enable_assignments': '{{enable_assignments}}',
				'enable_notifications': '{{enable_notifications}}',
				'enable_deadlines': '{{enable_deadlines}}'
			})
		elif template_type == TemplateType.E_COMMERCE:
			base_vars.update({
				'enable_payments': '{{enable_payments}}',
				'enable_shipping': '{{enable_shipping}}',
				'enable_inventory': '{{enable_inventory}}'
			})
		elif template_type == TemplateType.AI_ASSISTANT:
			base_vars.update({
				'ai_provider': '{{ai_provider}}',
				'enable_nlp': '{{enable_nlp}}',
				'knowledge_base': '{{knowledge_base}}'
			})
		
		return base_vars
	
	def _get_template_file_list(self, template_type: TemplateType) -> List[str]:
		"""Get list of files included in template"""
		base_files = [
			'main.apg.template',
			'README.md.template',
			'requirements.txt.template',
			'config.py.template'
		]
		
		# Add template-specific files
		if template_type in [TemplateType.E_COMMERCE, TemplateType.ENTERPRISE_DASHBOARD]:
			base_files.extend([
				'models/products.apg.template',
				'models/orders.apg.template',
				'templates/dashboard.html.template'
			])
		
		if template_type == TemplateType.MICROSERVICES:
			base_files.extend([
				'services/api_gateway.apg.template',
				'services/user_service.apg.template',
				'services/notification_service.apg.template',
				'docker-compose.yml.template'
			])
		
		if template_type == TemplateType.IOT_PLATFORM:
			base_files.extend([
				'devices/sensor_twin.apg.template',
				'analytics/data_processor.apg.template',
				'templates/device_dashboard.html.template'
			])
		
		return base_files
	
	def _get_template_apg_content(self, template_type: TemplateType) -> str:
		"""Get APG source content for template"""
		
		if template_type == TemplateType.BASIC_AGENT:
			return '''module {{project_name}} version {{version}} {
	description: "{{project_description}}";
	author: "{{author}}";
	license: "{{license}}";
}

agent BasicAgent {
	name: str = "{{project_name}} Agent";
	status: str = "inactive";
	counter: int = 0;
	
	initialize: () -> bool = {
		status = "active";
		counter = 0;
		return true;
	};
	
	process: () -> str = {
		if (status == "active") {
			counter = counter + 1;
			return "Processing request #" + str(counter);
		}
		return "Agent is inactive";
	};
	
	get_status: () -> dict = {
		return {
			"name": name,
			"status": status,
			"processed": counter,
			"timestamp": now()
		};
	};
}'''
		
		elif template_type == TemplateType.TASK_MANAGEMENT:
			return '''module {{project_name}} version {{version}} {
	description: "{{project_description}}";
	author: "{{author}}";
	license: "{{license}}";
}

agent TaskManagerAgent {
	name: str = "Task Manager";
	tasks: list[dict] = [];
	total_tasks: int = 0;
	completed_tasks: int = 0;
	active: bool = false;
	
	add_task: (title: str, priority: str, assignee: str) -> dict = {
		task = {
			"id": total_tasks + 1,
			"title": title,
			"priority": priority,
			"assignee": assignee,
			"status": "pending",
			"created_at": now(),
			"due_date": null
		};
		
		tasks.append(task);
		total_tasks = total_tasks + 1;
		
		// Send notification if enabled
		if ({{enable_notifications}}) {
			send_notification("task_created", task);
		}
		
		return task;
	};
	
	assign_task: (task_id: int, assignee: str) -> bool = {
		for (task in tasks) {
			if (task["id"] == task_id) {
				task["assignee"] = assignee;
				task["assigned_at"] = now();
				
				if ({{enable_notifications}}) {
					send_notification("task_assigned", task);
				}
				
				return true;
			}
		}
		return false;
	};
	
	complete_task: (task_id: int) -> bool = {
		for (task in tasks) {
			if (task["id"] == task_id) {
				task["status"] = "completed";
				task["completed_at"] = now();
				completed_tasks = completed_tasks + 1;
				
				if ({{enable_notifications}}) {
					send_notification("task_completed", task);
				}
				
				return true;
			}
		}
		return false;
	};
	
	get_task_stats: () -> dict = {
		return {
			"total": total_tasks,
			"completed": completed_tasks,
			"pending": total_tasks - completed_tasks,
			"completion_rate": completed_tasks / total_tasks * 100
		};
	};
}

workflow TaskWorkflow {
	name: str = "Task Processing Workflow";
	steps: list[str] = ["validate", "assign", "process", "review", "complete"];
	current_step: int = 0;
	task_id: int = 0;
	
	start_workflow: (task_id: int) -> bool = {
		this.task_id = task_id;
		current_step = 0;
		return true;
	};
	
	execute_next_step: () -> dict = {
		if (current_step < len(steps)) {
			step_name = steps[current_step];
			current_step = current_step + 1;
			
			return {
				"step": step_name,
				"completed": true,
				"progress": current_step / len(steps) * 100
			};
		}
		
		return {
			"error": "Workflow completed",
			"progress": 100
		};
	};
}

db TaskDatabase {
	url: "sqlite:///tasks.db";
	
	schema task_schema {
		table tasks {
			id serial [pk]
			title varchar(200) [not null]
			description text
			priority varchar(20) [default: "medium"]
			status varchar(20) [default: "pending"]
			assignee varchar(100)
			created_at timestamp [default: now()]
			due_date timestamp
			completed_at timestamp
			
			indexes {
				(status)
				(priority)
				(assignee)
				(created_at)
			}
		}
		
		table users {
			id serial [pk]
			username varchar(50) [unique, not null]
			email varchar(255) [unique, not null]
			full_name varchar(100)
			role varchar(50) [default: "user"]
			created_at timestamp [default: now()]
		}
		
		table task_assignments {
			id serial [pk]
			task_id int [ref: > tasks.id]
			user_id int [ref: > users.id]
			assigned_at timestamp [default: now()]
			assigned_by int [ref: > users.id]
		}
		
		{% if enable_notifications %}
		table notifications {
			id serial [pk]
			user_id int [ref: > users.id]
			type varchar(50) [not null]
			title varchar(200) [not null]
			message text [not null]
			read bool [default: false]
			created_at timestamp [default: now()]
		}
		{% endif %}
	}
}'''
		
		elif template_type == TemplateType.E_COMMERCE:
			return '''module {{project_name}} version {{version}} {
	description: "{{project_description}}";
	author: "{{author}}";
	license: "{{license}}";
}

agent ProductCatalogAgent {
	products: list[dict] = [];
	categories: list[dict] = [];
	total_products: int = 0;
	
	add_product: (name: str, description: str, price: float, category_id: int) -> dict = {
		product = {
			"id": total_products + 1,
			"name": name,
			"description": description,
			"price": price,
			"category_id": category_id,
			"stock": 0,
			"active": true,
			"created_at": now()
		};
		
		products.append(product);
		total_products = total_products + 1;
		
		return product;
	};
	
	update_stock: (product_id: int, quantity: int) -> bool = {
		for (product in products) {
			if (product["id"] == product_id) {
				product["stock"] = quantity;
				product["updated_at"] = now();
				return true;
			}
		}
		return false;
	};
	
	search_products: (query: str, category_id: int) -> list[dict] = {
		results = [];
		
		for (product in products) {
			if (product["active"]) {
				// Simple text search
				if (query == "" or query in product["name"] or query in product["description"]) {
					if (category_id == 0 or product["category_id"] == category_id) {
						results.append(product);
					}
				}
			}
		}
		
		return results;
	};
}

agent ShoppingCartAgent {
	carts: dict = {}; // user_id -> cart
	
	add_to_cart: (user_id: int, product_id: int, quantity: int) -> dict = {
		if (user_id not in carts) {
			carts[user_id] = {
				"items": [],
				"total": 0.0,
				"created_at": now()
			};
		}
		
		cart = carts[user_id];
		
		// Check if item already exists
		for (item in cart["items"]) {
			if (item["product_id"] == product_id) {
				item["quantity"] = item["quantity"] + quantity;
				return cart;
			}
		}
		
		// Add new item
		item = {
			"product_id": product_id,
			"quantity": quantity,
			"added_at": now()
		};
		
		cart["items"].append(item);
		cart["updated_at"] = now();
		
		return cart;
	};
	
	remove_from_cart: (user_id: int, product_id: int) -> bool = {
		if (user_id in carts) {
			cart = carts[user_id];
			cart["items"] = [item for item in cart["items"] if item["product_id"] != product_id];
			cart["updated_at"] = now();
			return true;
		}
		return false;
	};
	
	get_cart: (user_id: int) -> dict = {
		return carts.get(user_id, {"items": [], "total": 0.0});
	};
}

agent OrderProcessingAgent {
	orders: list[dict] = [];
	order_counter: int = 0;
	
	create_order: (user_id: int, cart_items: list[dict], shipping_address: dict) -> dict = {
		order = {
			"id": order_counter + 1,
			"user_id": user_id,
			"items": cart_items,
			"shipping_address": shipping_address,
			"status": "pending",
			"total_amount": 0.0,
			"created_at": now(),
			"estimated_delivery": null
		};
		
		// Calculate total
		total = 0.0;
		for (item in cart_items) {
			total = total + (item["price"] * item["quantity"]);
		}
		order["total_amount"] = total;
		
		orders.append(order);
		order_counter = order_counter + 1;
		
		return order;
	};
	
	update_order_status: (order_id: int, status: str) -> bool = {
		for (order in orders) {
			if (order["id"] == order_id) {
				order["status"] = status;
				order["updated_at"] = now();
				
				if (status == "shipped") {
					order["shipped_at"] = now();
				} else if (status == "delivered") {
					order["delivered_at"] = now();
				}
				
				return true;
			}
		}
		return false;
	};
	
	get_order_history: (user_id: int) -> list[dict] = {
		user_orders = [];
		for (order in orders) {
			if (order["user_id"] == user_id) {
				user_orders.append(order);
			}
		}
		return user_orders;
	};
}

db ECommerceDatabase {
	url: "postgresql://localhost:5432/{{project_name}}";
	
	schema ecommerce_schema {
		table categories {
			id serial [pk]
			name varchar(100) [not null]
			description text
			parent_id int [ref: > categories.id]
			active bool [default: true]
			created_at timestamp [default: now()]
		}
		
		table products {
			id serial [pk]
			name varchar(200) [not null]
			description text
			price decimal(10,2) [not null]
			category_id int [ref: > categories.id]
			stock int [default: 0]
			sku varchar(50) [unique]
			weight decimal(8,2)
			dimensions varchar(50)
			active bool [default: true]
			created_at timestamp [default: now()]
			updated_at timestamp [default: now()]
			
			indexes {
				(category_id)
				(sku) [unique]
				(active)
				(price)
			}
		}
		
		table customers {
			id serial [pk]
			email varchar(255) [unique, not null]
			password_hash varchar(255) [not null]
			first_name varchar(100)
			last_name varchar(100)
			phone varchar(20)
			created_at timestamp [default: now()]
			last_login timestamp
		}
		
		table addresses {
			id serial [pk]
			customer_id int [ref: > customers.id]
			type varchar(20) [note: "billing, shipping"]
			street_address varchar(255) [not null]
			city varchar(100) [not null]
			state varchar(100)
			postal_code varchar(20)
			country varchar(100) [not null]
			is_default bool [default: false]
		}
		
		table orders {
			id serial [pk]
			customer_id int [ref: > customers.id]
			status varchar(20) [default: "pending"]
			total_amount decimal(10,2) [not null]
			shipping_address_id int [ref: > addresses.id]
			billing_address_id int [ref: > addresses.id]
			payment_method varchar(50)
			payment_status varchar(20) [default: "pending"]
			created_at timestamp [default: now()]
			shipped_at timestamp
			delivered_at timestamp
			
			indexes {
				(customer_id)
				(status)
				(created_at)
			}
		}
		
		table order_items {
			id serial [pk]
			order_id int [ref: > orders.id]
			product_id int [ref: > products.id]
			quantity int [not null]
			unit_price decimal(10,2) [not null]
			total_price decimal(10,2) [not null]
		}
		
		{% if enable_inventory %}
		table inventory_transactions {
			id serial [pk]
			product_id int [ref: > products.id]
			type varchar(20) [note: "in, out, adjustment"]
			quantity int [not null]
			reference_id int [note: "order_id or other reference"]
			notes text
			created_at timestamp [default: now()]
		}
		{% endif %}
	}
}'''
		
		elif template_type == TemplateType.AI_ASSISTANT:
			return '''module {{project_name}} version {{version}} {
	description: "{{project_description}}";
	author: "{{author}}";
	license: "{{license}}";
}

agent ConversationAgent {
	conversations: dict = {}; // user_id -> conversation history
	knowledge_base: list[dict] = [];
	ai_provider: str = "{{ai_provider}}";
	
	start_conversation: (user_id: int) -> dict = {
		if (user_id not in conversations) {
			conversations[user_id] = {
				"messages": [],
				"context": {},
				"created_at": now()
			};
		}
		
		return conversations[user_id];
	};
	
	process_message: (user_id: int, message: str) -> dict = {
		conversation = conversations.get(user_id, this.start_conversation(user_id));
		
		// Add user message to history
		user_msg = {
			"role": "user",
			"content": message,
			"timestamp": now()
		};
		conversation["messages"].append(user_msg);
		
		// Process with AI
		response = this.generate_response(conversation, message);
		
		// Add assistant response
		assistant_msg = {
			"role": "assistant", 
			"content": response,
			"timestamp": now()
		};
		conversation["messages"].append(assistant_msg);
		
		return {
			"response": response,
			"conversation_id": user_id,
			"message_count": len(conversation["messages"])
		};
	};
	
	generate_response: (conversation: dict, message: str) -> str = {
		// Simple rule-based responses (would integrate with actual AI)
		message_lower = message.lower();
		
		if ("hello" in message_lower or "hi" in message_lower) {
			return "Hello! How can I help you today?";
		} else if ("help" in message_lower) {
			return "I'm here to assist you. You can ask me questions or request help with various tasks.";
		} else if ("weather" in message_lower) {
			return "I'd be happy to help with weather information. Could you specify your location?";
		} else if ("time" in message_lower) {
			return "The current time is " + str(now());
		}
		
		// Search knowledge base
		relevant_info = this.search_knowledge_base(message);
		if (relevant_info) {
			return "Based on my knowledge: " + relevant_info["content"];
		}
		
		return "I understand you're asking about: " + message + ". Let me help you with that.";
	};
	
	search_knowledge_base: (query: str) -> dict = {
		// Simple keyword matching (would use vector search in production)
		query_lower = query.lower();
		
		for (item in knowledge_base) {
			if (query_lower in item["title"].lower() or query_lower in item["content"].lower()) {
				return item;
			}
		}
		
		return null;
	};
	
	add_knowledge: (title: str, content: str, category: str) -> dict = {
		knowledge_item = {
			"id": len(knowledge_base) + 1,
			"title": title,
			"content": content,
			"category": category,
			"created_at": now(),
			"usage_count": 0
		};
		
		knowledge_base.append(knowledge_item);
		return knowledge_item;
	};
}

agent IntentClassifierAgent {
	intents: dict = {
		"greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
		"question": ["what", "how", "when", "where", "why", "who"],
		"request": ["please", "can you", "could you", "would you"],
		"complaint": ["problem", "issue", "error", "bug", "wrong"]
	};
	
	classify_intent: (message: str) -> dict = {
		message_lower = message.lower();
		scores = {};
		
		for (intent, keywords in intents.items()) {
			score = 0;
			for (keyword in keywords) {
				if (keyword in message_lower) {
					score = score + 1;
				}
			}
			scores[intent] = score;
		}
		
		// Find best match
		best_intent = "unknown";
		best_score = 0;
		
		for (intent, score in scores.items()) {
			if (score > best_score) {
				best_intent = intent;
				best_score = score;
			}
		}
		
		return {
			"intent": best_intent,
			"confidence": best_score / len(message.split()),
			"all_scores": scores
		};
	};
}

db AssistantDatabase {
	url: "postgresql://localhost:5432/{{project_name}}";
	
	schema assistant_schema {
		table users {
			id serial [pk]
			username varchar(50) [unique, not null]
			email varchar(255) [unique, not null]
			preferences json
			created_at timestamp [default: now()]
			last_active timestamp
		}
		
		table conversations {
			id serial [pk]
			user_id int [ref: > users.id]
			title varchar(200)
			created_at timestamp [default: now()]
			updated_at timestamp [default: now()]
			message_count int [default: 0]
		}
		
		table messages {
			id serial [pk]
			conversation_id int [ref: > conversations.id]
			role varchar(20) [not null, note: "user, assistant, system"]
			content text [not null]
			intent varchar(50)
			confidence float
			created_at timestamp [default: now()]
			
			indexes {
				(conversation_id)
				(role)
				(created_at)
			}
		}
		
		table knowledge_base {
			id serial [pk]
			title varchar(200) [not null]
			content text [not null]
			category varchar(100)
			tags varchar(500)
			
			{% if enable_nlp %}
			// Vector embeddings for semantic search
			content_embedding vector(1536) [dimensions: 1536]
			title_embedding vector(1536) [dimensions: 1536]
			{% endif %}
			
			usage_count int [default: 0]
			created_at timestamp [default: now()]
			updated_at timestamp [default: now()]
			
			indexes {
				(category)
				(usage_count)
			}
			
			{% if enable_nlp %}
			vector_index idx_content_search on knowledge_base (content_embedding) [
				method: hnsw,
				distance: cosine,
				dimensions: 1536
			]
			{% endif %}
		}
		
		table user_feedback {
			id serial [pk]
			user_id int [ref: > users.id]
			message_id int [ref: > messages.id]
			feedback_type varchar(20) [note: "helpful, not_helpful, incorrect"]
			rating int [note: "1-5 scale"]
			comment text
			created_at timestamp [default: now()]
		}
	}
}'''
		
		else:
			# Default basic template
			return self._get_template_apg_content(TemplateType.BASIC_AGENT)
	
	def _get_template_additional_files(self, template_type: TemplateType) -> Dict[str, str]:
		"""Get additional template files content"""
		files = {}
		
		# README template
		files["README.md.template"] = f'''# {{{{project_name}}}}

{{{{project_description}}}}

## Features

{self._get_feature_list(template_type)}

## Installation

1. Install APG compiler:
   ```bash
   pip install apg-language
   ```

2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Compile APG source:
   ```bash
   apg compile main.apg
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the application:
   Open http://localhost:8080

## Usage

{self._get_usage_instructions(template_type)}

## Configuration

Edit `config.py` to customize:
- Database connection
- Authentication settings
- Feature flags

## Development

- APG source: `main.apg`
- Generated code: `generated/`
- Templates: `templates/`
- Static files: `static/`

## License

{{{{license}}}} - see LICENSE file for details.

---

Generated with APG (Application Programming Generation) language.
'''
		
		# Requirements template
		files["requirements.txt.template"] = '''# APG Generated Application Requirements
Flask-AppBuilder>=4.3.0
Flask>=2.3.0
Flask-SQLAlchemy>=3.0.0
SQLAlchemy>=2.0.0
WTForms>=3.0.0
Werkzeug>=2.3.0

# Template-specific requirements
''' + self._get_template_requirements(template_type)
		
		# Configuration template
		files["config.py.template"] = '''"""
Application Configuration
========================

Configuration for {{project_name}} Flask-AppBuilder application.
"""

import os
from flask_appbuilder.security.manager import AUTH_DB

basedir = os.path.abspath(os.path.dirname(__file__))

# Security
SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

# Database
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')

# Flask-AppBuilder
APP_NAME = "{{project_name}}"
AUTH_TYPE = AUTH_DB

# Template-specific configuration
''' + self._get_template_config_extra(template_type)
		
		return files
	
	def _get_feature_list(self, template_type: TemplateType) -> str:
		"""Get feature list for README"""
		info = get_template_info(template_type)
		features = info.get('features', [])
		
		if not features:
			return "- Basic agent functionality\n- Web dashboard\n- Database integration"
		
		return '\n'.join(f"- {feature}" for feature in features)
	
	def _get_usage_instructions(self, template_type: TemplateType) -> str:
		"""Get usage instructions for README"""
		if template_type == TemplateType.TASK_MANAGEMENT:
			return '''### Task Management

1. **Add Tasks**: Use the web interface or API to create new tasks
2. **Assign Tasks**: Assign tasks to team members
3. **Track Progress**: Monitor task completion and statistics
4. **Notifications**: Receive updates on task changes

### API Endpoints

- `POST /api/tasks` - Create new task
- `GET /api/tasks` - List all tasks
- `PUT /api/tasks/{id}` - Update task
- `DELETE /api/tasks/{id}` - Delete task
'''
		
		elif template_type == TemplateType.E_COMMERCE:
			return '''### E-Commerce Platform

1. **Product Management**: Add and manage product catalog
2. **Order Processing**: Handle customer orders and payments
3. **Inventory Tracking**: Monitor stock levels
4. **Customer Management**: Manage customer accounts and addresses

### Key Workflows

- Product catalog management
- Shopping cart functionality
- Order processing pipeline
- Inventory management
'''
		
		else:
			return '''### Basic Usage

1. **Agent Dashboard**: Monitor agent status and activity
2. **Method Execution**: Execute agent methods via web interface
3. **Real-time Updates**: View live status and logs
4. **Configuration**: Customize agent behavior through settings
'''
	
	def _get_template_requirements(self, template_type: TemplateType) -> str:
		"""Get template-specific requirements"""
		if template_type == TemplateType.AI_ASSISTANT:
			return '''
# AI/ML requirements
openai>=1.0.0
transformers>=4.0.0
sentence-transformers>=2.0.0
'''
		elif template_type == TemplateType.E_COMMERCE:
			return '''
# Payment processing
stripe>=5.0.0

# Image processing
Pillow>=8.0.0
'''
		elif template_type == TemplateType.IOT_PLATFORM:
			return '''
# IoT and real-time features
paho-mqtt>=1.6.0
redis>=4.0.0
celery>=5.0.0
'''
		else:
			return '''
# Basic web application requirements
requests>=2.28.0
'''
	
	def _get_template_config_extra(self, template_type: TemplateType) -> str:
		"""Get template-specific configuration"""
		if template_type == TemplateType.AI_ASSISTANT:
			return '''
# AI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
AI_MODEL = os.environ.get('AI_MODEL', 'gpt-3.5-turbo')
ENABLE_NLP = {{enable_nlp}}
'''
		elif template_type == TemplateType.E_COMMERCE:
			return '''
# E-Commerce Configuration
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
ENABLE_PAYMENTS = {{enable_payments}}
ENABLE_INVENTORY = {{enable_inventory}}
'''
		else:
			return '''
# Application-specific configuration
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
'''
	
	def get_template(self, template_type: TemplateType) -> Optional[Dict[str, Any]]:
		"""Get template by type"""
		template_dir = self.templates_dir / template_type.value
		config_file = template_dir / "template.json"
		
		if not config_file.exists():
			self._create_builtin_template(template_type)
		
		try:
			with open(config_file, 'r') as f:
				config = json.load(f)
			
			# Load template files
			template_files = {}
			for filename in config.get('files', []):
				file_path = template_dir / filename
				if file_path.exists():
					with open(file_path, 'r') as f:
						template_files[filename] = f.read()
			
			config['template_files'] = template_files
			config['template_type'] = template_type
			
			return config
			
		except Exception as e:
			print(f"Error loading template {template_type.value}: {e}")
			return None
	
	def list_templates(self) -> List[Dict[str, Any]]:
		"""List all available templates"""
		templates = []
		
		for template_type in TemplateType:
			template_info = self.get_template(template_type)
			if template_info:
				templates.append({
					'type': template_type.value,
					'name': template_info.get('name'),
					'description': template_info.get('description'),
					'complexity': template_info.get('complexity'),
					'features': template_info.get('features', []),
					'use_cases': template_info.get('use_cases', [])
				})
		
		return templates
	
	def validate_template(self, template_type: TemplateType) -> List[str]:
		"""Validate template completeness"""
		errors = []
		
		template_dir = self.templates_dir / template_type.value
		
		# Check template directory exists
		if not template_dir.exists():
			errors.append(f"Template directory missing: {template_dir}")
			return errors
		
		# Check required files
		required_files = ["template.json", "main.apg.template"]
		for filename in required_files:
			file_path = template_dir / filename
			if not file_path.exists():
				errors.append(f"Missing required file: {filename}")
		
		# Validate template.json
		config_file = template_dir / "template.json"
		if config_file.exists():
			try:
				with open(config_file, 'r') as f:
					config = json.load(f)
				
				required_keys = ["name", "description", "variables", "files"]
				for key in required_keys:
					if key not in config:
						errors.append(f"Missing key in template.json: {key}")
						
			except json.JSONDecodeError as e:
				errors.append(f"Invalid JSON in template.json: {e}")
		
		return errors