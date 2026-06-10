# APG Studio — User Guide

## Overview

APG Studio is a web-based development environment for the Application Programming Generation platform. It provides:

1. **Landing Page** (`/studio/`) — interactive overview of APG's 265 capabilities, architecture, and interoperability
2. **Compositor** (`/studio/compositor`) — full web IDE for writing, compiling, and downloading APG programs

## Running the Studio

```bash
# From the repo root
uv run python capabilities/composition/studio/app.py

# Or with env vars
STUDIO_PORT=5100 uv run python capabilities/composition/studio/app.py
```

Then open `http://localhost:5100/studio/`

## Compositor Features

### Editor
- Full APG syntax highlighting (keywords, types, properties, strings)
- Tab indent support
- `⌘↵` (or `Ctrl+Enter`) to compile instantly
- Line count in status bar

### Capability Browser (left sidebar)
- **Capabilities tab**: searchable tree of all 265 APG capabilities grouped by domain
  - Click any capability to insert a `uses: [cap_id]` reference at cursor
- **Examples tab**: load real example programs (marketplace, AI agents, OSINT, etc.)
- **Templates tab**: insert pre-built templates (fintech, healthcare, approval workflow, connector)

### Compilation
- Click **Compile** or press `⌘↵`
- Generated files appear as tabs in the right output panel: `service.py`, `models.py`, `api.py`, etc.
- Errors and warnings shown in the Errors tab with line numbers
- **Copy** button copies the active file content to clipboard

### Download
- Click **Download** to get all generated files as a ZIP archive (`apg_generated.zip`)
- Fallback: downloads the first file as `.py` if ZIP fails

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/studio/` | Landing page |
| GET | `/studio/compositor` | Web IDE |
| GET | `/studio/api/capabilities` | List all 265 capabilities |
| GET | `/studio/api/examples` | List available examples |
| GET | `/studio/api/examples/{name}` | Get example source |
| GET | `/studio/api/templates` | List code templates |
| GET | `/studio/api/templates/{name}` | Get template source |
| POST | `/studio/api/compile` | Compile APG source → Python files |
| POST | `/studio/api/download` | Download files as ZIP |
| GET | `/studio/api/health` | Health check |

### Compile request

```json
POST /studio/api/compile
{
  "source": "capability Foo { ... }",
  "filename": "foo.apg"
}
```

Response:
```json
{
  "success": true,
  "files": {
    "service.py": "...",
    "models.py": "...",
    "api.py": "...",
    "views.py": "..."
  },
  "warnings": [],
  "file_count": 7
}
```

## Integration with Flask-AppBuilder

To embed Studio in a full APG application:

```python
from capabilities.composition.studio.api import studio_api
app.register_blueprint(studio_api)
```

The blueprint registers at `/studio` prefix and serves static files from `capabilities/composition/studio/static/`.
