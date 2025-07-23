# APG Language Support for VS Code

Full-featured language support for APG (Application Programming Generation) language in Visual Studio Code.

## Features

### 🎨 **Syntax Highlighting**
- Rich syntax highlighting for all APG language constructs
- Support for agents, digital twins, workflows, databases, and more
- Specialized highlighting for DBML database definitions
- Vector storage and AI/ML specific syntax highlighting

### 🧠 **IntelliSense & Code Completion**
- Context-aware code completion
- Entity-specific suggestions (agents, digital twins, workflows)
- Database schema completion with DBML support
- Built-in function and type completion
- Smart property and method suggestions

### 🔍 **Language Server Integration**
- Real-time error checking and validation
- Hover information for symbols and keywords
- Go to definition and find references
- Document symbols for code navigation
- Semantic analysis and type checking

### ⚡ **Code Snippets**
- Comprehensive snippet library for rapid development
- Module, entity, and method templates
- Database table and schema snippets
- Vector storage and AI/ML templates
- Workflow and automation snippets

### 🔧 **Build & Compilation**
- One-click compilation to Flask-AppBuilder applications
- Project-wide build support
- Real-time syntax validation
- Integration with APG CLI tools

### 🚀 **Development Workflow**
- Create new APG projects directly from VS Code
- Run generated Flask-AppBuilder applications
- Live preview of APG code
- Integrated terminal for APG commands

## Installation

1. **Install the Extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "APG Language Support"
   - Click Install

2. **Install APG Language Tools**:
   ```bash
   pip install apg-language
   ```

3. **Start Language Server** (automatic):
   The extension will automatically start the APG Language Server when you open APG files.

## Quick Start

### Create New Project
1. Open Command Palette (Ctrl+Shift+P)
2. Run "APG: Create New Project"
3. Enter project name and select template
4. Start coding!

### Basic APG File
Create a file with `.apg` extension:

```apg
module hello_world version 1.0.0 {
    description: "My first APG application";
    author: "Your Name";
}

agent HelloAgent {
    name: str = "Hello Agent";
    message: str = "Hello from APG!";
    
    greet: (visitor: str) -> str = {
        return message + " Welcome, " + visitor + "!";
    };
    
    process: () -> str = {
        return greet("World");
    };
}
```

### Compile and Run
1. **Compile**: Ctrl+Shift+B or "APG: Compile Current File"
2. **Run**: Ctrl+F5 or "APG: Run Generated Application"
3. **Access**: Open http://localhost:8080 in your browser

## Language Features

### Entity Types
- **Agents**: Autonomous processing units
- **Digital Twins**: Real-world entity representations
- **Workflows**: Process automation and orchestration
- **Databases**: Schema definition with DBML integration
- **APIs**: RESTful service definitions
- **Forms**: UI form generation

### Database Support
- Full DBML (Database Markup Language) integration
- Vector storage for AI/ML applications (vector, embedding, halfvec, sparsevec)
- Advanced features: triggers, procedures, functions, views
- Multiple database backends (PostgreSQL, MySQL, SQLite)

### Generated Applications
- Professional Flask-AppBuilder web applications
- Interactive dashboards and control panels
- RESTful APIs with authentication
- Database management interfaces
- Real-time monitoring and logging

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| APG: Compile Current File | Ctrl+Shift+B | Compile the active APG file |
| APG: Compile Project | - | Build entire APG project |
| APG: Run Generated App | Ctrl+F5 | Start the generated Flask app |
| APG: Validate Syntax | Ctrl+Shift+V | Check APG file syntax |
| APG: Show Preview | - | Display APG code preview |
| APG: Create New Project | - | Initialize new APG project |
| APG: Restart Language Server | - | Restart the language server |

## Configuration

Configure the extension in VS Code settings:

```json
{
    "apg.languageServer.enabled": true,
    "apg.languageServer.host": "127.0.0.1",
    "apg.languageServer.port": 2087,
    "apg.compiler.target": "flask-appbuilder",
    "apg.compiler.outputDir": "generated",
    "apg.validation.enableRealTime": true,
    "apg.completion.enabled": true,
    "apg.hover.enabled": true,
    "apg.symbols.enabled": true
}
```

## Themes

The extension includes APG-optimized color themes:
- **APG Dark**: Dark theme with APG-specific highlighting
- **APG Light**: Light theme optimized for APG syntax

Select themes from: File → Preferences → Color Theme

## Troubleshooting

### Language Server Not Starting
1. Ensure APG tools are installed: `pip install apg-language`
2. Check that `apg-language-server` is in your PATH
3. Restart VS Code
4. Use "APG: Restart Language Server" command

### Compilation Errors
1. Verify APG CLI is installed: `apg --version`
2. Check APG project structure (apg.json file)
3. Review output in APG Language output panel

### Missing Features
1. Update to latest extension version
2. Ensure Language Server is connected (check status bar)
3. Restart Language Server if needed

## Examples

### Complete Agent Example
```apg
module task_manager version 1.0.0 {
    description: "Task management system";
    author: "Developer";
}

agent TaskManager {
    tasks: list[dict] = [];
    completed: int = 0;
    
    add_task: (title: str, priority: str) -> dict = {
        task = {
            "id": len(tasks) + 1,
            "title": title,
            "priority": priority,
            "status": "pending",
            "created": now()
        };
        
        tasks.append(task);
        return task;
    };
    
    complete_task: (task_id: int) -> bool = {
        for (task in tasks) {
            if (task["id"] == task_id) {
                task["status"] = "completed";
                completed = completed + 1;
                return true;
            }
        }
        return false;
    };
    
    get_stats: () -> dict = {
        return {
            "total": len(tasks),
            "completed": completed,
            "pending": len(tasks) - completed
        };
    };
}
```

### Database with Vector Storage
```apg
db RecommendationDB {
    url: "postgresql://localhost:5432/recommendations";
    
    schema ai_schema {
        table products {
            id serial [pk]
            name varchar(200) [not null]
            description text
            
            // Vector embeddings for ML
            content_embedding vector(1536) [dimensions: 1536, normalized]
            image_features halfvec(512) [dimensions: 512]
            
            created_at timestamp [default: now()]
            
            // Vector similarity index
            vector_index idx_content_similarity on products (content_embedding) [
                method: hnsw,
                distance: cosine,
                dimensions: 1536
            ]
        }
        
        // Auto-update embeddings
        trigger update_embeddings after insert or update on products {
            begin
                execute procedure refresh_embeddings(NEW.id);
            end
        }
        
        procedure refresh_embeddings(in product_id int) [language: plpgsql] {
            begin
                // Update embedding logic
                update products 
                set content_embedding = generate_embedding(name || ' ' || description)
                where id = product_id;
            end
        }
    }
}
```

## Support

- **Documentation**: https://apg-lang.org/docs
- **GitHub**: https://github.com/apg-lang/vscode-extension
- **Issues**: https://github.com/apg-lang/vscode-extension/issues
- **Community**: https://discord.gg/apg-lang

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Enjoy developing with APG! 🚀**