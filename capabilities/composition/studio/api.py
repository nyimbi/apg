"""APG Studio — backend API for landing page and compositor."""
from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file, send_from_directory

_log = logging.getLogger(__name__)

studio_api = Blueprint(
    "studio_api", __name__,
    url_prefix="/studio",
    static_folder=str(Path(__file__).parent / "static"),
    static_url_path="/static",
    template_folder=str(Path(__file__).parent / "templates"),
)

_MANIFEST_PATH = Path(__file__).parent.parent.parent.parent / "capabilities" / "MANIFEST.json"
_EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples"

_MANIFEST_CACHE: dict[str, Any] | None = None


def _manifest() -> dict[str, Any]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        with open(_MANIFEST_PATH) as f:
            _MANIFEST_CACHE = json.load(f)
    return _MANIFEST_CACHE


# ── Landing page ──────────────────────────────────────────────────

@studio_api.get("/")
@studio_api.get("")
def landing():
    from flask import render_template
    m = _manifest()
    caps = list(m["capabilities"].values())

    domains = {}
    for c in caps:
        d = c.get("domain", "other")
        domains[d] = domains.get(d, 0) + 1

    DOMAIN_META = {
        "fintech": ("🏦", "Payments, Banking, Fraud"),
        "intel": ("🕵", "Intelligence, Surveillance"),
        "healthcare": ("🏥", "EMR, Clinical, Telemedicine"),
        "pharma": ("💊", "GxP, LIMS, QMS, CTM"),
        "common": ("⚙", "Infrastructure, Auth, ML"),
        "composition": ("🔗", "Gateway, Orchestration"),
        "bia": ("📊", "Analytics, Forecasting"),
        "government": ("🏛", "Tax, Licensing, Case"),
        "realestate": ("🏠", "Property, Tenancy"),
        "telecom": ("📡", "Network, Billing, OSS"),
        "transport": ("🚚", "Fleet, Dispatch, Routing"),
        "retail": ("🛍", "POS, Loyalty, Promotions"),
        "education": ("🎓", "LMS, Timetable"),
        "energy": ("⚡", "Grid, Metering, Renewables"),
        "mining": ("⛏", "Exploration, Production"),
        "crm": ("👥", "Advanced CRM, CPQ"),
        "grc": ("🛡", "Risk, Compliance, Audit"),
        "hcm": ("👔", "Payroll, Time & Attendance"),
        "fin": ("💰", "Accounts, Budgeting"),
        "loc": ("📍", "Local Services"),
        "mob": ("📱", "Mobile, MDM"),
        "ppm": ("📋", "Project Portfolio"),
        "ecd": ("🏗", "Engineering Design"),
        "pde": ("🔧", "Product Development"),
        "scm": ("📦", "Supply Chain"),
        "int": ("🔌", "Integration Platform"),
        "eam": ("🔩", "Asset Management"),
    }

    domain_list = []
    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        icon, sample = DOMAIN_META.get(d, ("📦", d.title()))
        domain_list.append({"label": d.title(), "icon": icon, "sample": sample, "count": count})

    stats = {
        "capabilities": len(caps),
        "domains": len(domains),
        "tests": 1261,
        "methods": f"{sum(c.get('service_method_count', 0) for c in caps) // len(caps)}+",
        "connectors": 6,
    }

    return render_template("landing.html", domains=domain_list, stats=stats)


# ── Compositor ────────────────────────────────────────────────────

@studio_api.get("/compositor")
def compositor():
    from flask import render_template
    return render_template("compositor.html")


# ── Capabilities API ──────────────────────────────────────────────

@studio_api.get("/api/capabilities")
def list_capabilities():
    m = _manifest()
    caps = list(m["capabilities"].values())
    simplified = [
        {
            "id": c["id"],
            "display_name": c.get("display_name", c["id"]),
            "domain": c.get("domain", ""),
            "description": c.get("description", "")[:120],
            "provides": c.get("provides", [])[:5],
            "requires": c.get("requires", [])[:5],
            "service_method_count": c.get("service_method_count", 0),
        }
        for c in caps
    ]
    return jsonify({"capabilities": simplified, "total": len(simplified)})


# ── Examples API ──────────────────────────────────────────────────

_EXAMPLE_REGISTRY = [
    {"name": "marketplace_platform", "title": "Marketplace Platform", "icon": "🛍", "file": "marketplace_platform.apg"},
    {"name": "ai_agent_systems", "title": "AI Agent Systems", "icon": "🤖", "file": "ai_agent_systems.apg"},
    {"name": "microservices_architecture", "title": "Microservices Architecture", "icon": "🔗", "file": "microservices_architecture.apg"},
    {"name": "osint_intelligence", "title": "OSINT Intelligence", "icon": "🕵", "file": "osint_intelligence.apg"},
    {"name": "production_line_monitoring", "title": "Production Line Monitoring", "icon": "⚙", "file": "production_line_monitoring.apg"},
    {"name": "digital_twin_examples", "title": "Digital Twin", "icon": "🔮", "file": "digital_twin_examples.apg"},
]

@studio_api.get("/api/examples")
def list_examples():
    available = []
    for ex in _EXAMPLE_REGISTRY:
        path = _EXAMPLES_DIR / ex["file"]
        if path.exists():
            available.append({"name": ex["name"], "title": ex["title"], "icon": ex["icon"]})
    return jsonify({"examples": available})


@studio_api.get("/api/examples/<name>")
def get_example(name: str):
    for ex in _EXAMPLE_REGISTRY:
        if ex["name"] == name:
            path = _EXAMPLES_DIR / ex["file"]
            if path.exists():
                return jsonify({"source": path.read_text(), "filename": ex["file"]})
    return jsonify({"error": "example not found"}), 404


# ── Templates API ─────────────────────────────────────────────────

_TEMPLATES = {
    "capability": {
        "title": "Basic Capability",
        "icon": "🧩",
        "source": """\
// APG Capability Template
capability {CapabilityName} {
  description: "{A concise description}";
  domain: {your_domain};

  contract: {
    operations: [
      create_{entity}, read_{entity},
      update_{entity}, delete_{entity},
      list_{entities}, search_{entities}
    ];
    governance: [
      tenant_context_required,
      actor_id_required,
      write_requires_policy
    ];
  };

  entity {Entity} {
    id: uuid;
    name: string;
    tenant_id: string;
    created_at: datetime;
  }
}
""",
    },
    "workflow": {
        "title": "Approval Workflow",
        "icon": "🔄",
        "source": """\
// APG Approval Workflow Template
workflow ApprovalFlow {
  states: [draft, pending_review, approved, rejected];
  initial: draft;

  human_task ReviewRequest {
    assignee_role: approver;
    sla: 24h;
    form: {
      fields: [decision, comments];
      decision: enum [approve, reject];
    };
  }

  transition draft -> pending_review {
    on: submitted;
    emit: review_requested;
  }

  transition pending_review -> approved {
    guard: decision == approve;
    on: task_completed;
    emit: request_approved;
  }

  transition pending_review -> rejected {
    guard: decision == reject;
    on: task_completed;
    emit: request_rejected;
  }
}
""",
    },
    "connector": {
        "title": "External Connector",
        "icon": "🔌",
        "source": """\
// APG External Connector Template
connector {ConnectorName} {
  provider: {provider_name};
  environment: sandbox;
  auth: oauth2;

  config: {
    client_id: $ENV_CLIENT_ID;
    client_secret: $ENV_CLIENT_SECRET;
    base_url: "https://api.{provider}.com";
  };

  operations: [
    check_status,
    create_transaction,
    get_transaction,
    list_transactions,
    reverse_transaction
  ];
}
""",
    },
    "fintech": {
        "title": "Fintech Service",
        "icon": "🏦",
        "source": """\
// APG Fintech Service Template
capability MobilePayments {
  description: "Mobile money payment processing";
  domain: fintech;

  contract: {
    operations: [
      initiate_payment,
      check_payment_status,
      list_transactions,
      reconcile,
      generate_statement
    ];
    governance: [
      tenant_context_required,
      pci_scope_enforced,
      amount_positive,
      phone_number_valid,
      daily_limit_enforced
    ];
    compliance: [PCI_DSS, KYC, AML];
  };

  uses: [vault, nats, fintech_frd, fintech_aml];

  entity Payment {
    id: uuid;
    amount: float;
    currency: string;
    phone: string;
    reference: string;
    status: enum [pending, completed, failed, reversed];
    provider: enum [mpesa, equity, kcb, stripe];
    created_at: datetime;
    tenant_id: string;
  }

  screen PaymentDashboard {
    layout: analytics_grid;
    widgets: [
      daily_volume, success_rate,
      failed_transactions, reconciliation_status
    ];
  }
}
""",
    },
    "healthcare": {
        "title": "Healthcare Capability",
        "icon": "🏥",
        "source": """\
// APG Healthcare Capability Template
capability PatientManagement {
  description: "HIPAA-compliant patient record management";
  domain: healthcare;

  contract: {
    operations: [
      register_patient,
      update_demographics,
      record_encounter,
      get_patient_history,
      export_fhir_r4,
      schedule_appointment
    ];
    governance: [
      tenant_context_required,
      hipaa_phi_access_logged,
      minimum_necessary_enforced,
      baa_required_for_sharing
    ];
    compliance: [HIPAA, HITECH, FHIR_R4];
  };

  uses: [phi, healthcare_emr];

  entity Patient {
    id: uuid;
    mrn: string;
    first_name: string;
    last_name: string;
    date_of_birth: date;
    gender: enum [M, F, other, unknown];
    tenant_id: string;
  }
}
""",
    },
}

@studio_api.get("/api/templates")
def list_templates():
    return jsonify({
        "templates": [
            {"name": k, "title": v["title"], "icon": v["icon"]}
            for k, v in _TEMPLATES.items()
        ]
    })


@studio_api.get("/api/templates/<name>")
def get_template(name: str):
    tmpl = _TEMPLATES.get(name)
    if not tmpl:
        return jsonify({"error": "template not found"}), 404
    return jsonify({"source": tmpl["source"], "filename": f"{name}_template.apg"})


# ── Compile API ───────────────────────────────────────────────────

@studio_api.post("/api/compile")
def compile_source():
    body = request.get_json(force=True) or {}
    source: str = body.get("source", "")
    filename: str = body.get("filename", "untitled.apg") or "untitled.apg"

    if not source.strip():
        return jsonify({"success": False, "errors": [{"message": "Empty source", "line": None}], "warnings": []})

    try:
        import sys
        repo_root = Path(__file__).parent.parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from compiler.compiler import APGCompiler  # type: ignore[import]
        result = APGCompiler().compile_string(source, filename)

        if result.success:
            # Filter to key generated files
            files = {}
            PRIORITY_ORDER = ["app.py", "service.py", "models.py", "views.py", "api.py", "blueprint.py",
                              "capability_contract.py", "ai_agents.py", "ai_agent_teams.py"]
            generated = dict(result.generated_files)
            for name in PRIORITY_ORDER:
                if name in generated:
                    files[name] = generated.pop(name)
            # Add remaining files (SQL, tests, etc.)
            for name, content in sorted(generated.items()):
                if not name.endswith((".pyc",)):
                    files[name] = content

            warnings = [{"message": str(w), "line": getattr(w, "line", None)} for w in (result.warnings or [])]
            return jsonify({"success": True, "files": files, "warnings": warnings, "file_count": len(files)})
        else:
            errors = [{"message": str(e), "line": getattr(e, "line", None)} for e in (result.errors or [])]
            warnings = [{"message": str(w), "line": getattr(w, "line", None)} for w in (result.warnings or [])]
            return jsonify({"success": False, "errors": errors, "warnings": warnings, "files": {}})

    except ImportError as e:
        _log.exception("compiler not available")
        return jsonify({"success": False, "errors": [{"message": f"Compiler not available: {e}", "line": None}], "warnings": []})
    except Exception as e:
        _log.exception("compilation error")
        return jsonify({"success": False, "errors": [{"message": str(e), "line": None}], "warnings": []})


# ── Download API ──────────────────────────────────────────────────

@studio_api.post("/api/download")
def download_zip():
    body = request.get_json(force=True) or {}
    files: dict[str, str] = body.get("files", {})

    if not files:
        return jsonify({"error": "no files"}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(f"apg_generated/{name}", content)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="apg_generated.zip")


# ── Health ────────────────────────────────────────────────────────

@studio_api.get("/api/health")
def health():
    m = _manifest()
    return jsonify({"status": "ok", "capabilities": len(m.get("capabilities", {}))})
