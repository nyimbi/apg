"""CLI coverage for executable capability contract commands."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "cli.py"
SPEC = importlib.util.spec_from_file_location("apg_root_cli", CLI_PATH)
assert SPEC is not None
assert SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["apg_root_cli"] = CLI_MODULE
SPEC.loader.exec_module(CLI_MODULE)

APGCLICommands = CLI_MODULE.APGCLICommands
create_parser = CLI_MODULE.create_parser


def test_cli_lists_capability_contracts(capsys):
	cli = APGCLICommands()

	assert cli.list_capability_contracts() is True
	output = capsys.readouterr().out

	assert "Capability contracts:" in output
	assert "composition_events" in output
	assert "fintech_gateway" in output


def test_cli_lists_capability_contracts_as_json(capsys):
	cli = APGCLICommands()

	assert cli.list_capability_contracts(output_json=True) is True
	output = capsys.readouterr().out
	payload = json.loads(output)

	assert len(payload) >= 100
	assert any(item["capability"] == "composition_events" for item in payload)
	assert all(item["routes"] > 0 for item in payload)
	assert all(item["rules"] > 0 for item in payload)


def test_cli_validates_capability_contracts(capsys):
	cli = APGCLICommands()

	assert cli.validate_capability_contracts() is True
	output = capsys.readouterr().out

	assert "Validated" in output
	assert "capability contracts" in output


def test_parser_accepts_capability_contract_actions():
	parser = create_parser()

	contracts_args = parser.parse_args(["capabilities", "contracts", "--json"])
	validate_args = parser.parse_args(["capabilities", "validate-contracts"])

	assert contracts_args.command == "capabilities"
	assert contracts_args.capability_action == "contracts"
	assert contracts_args.json is True
	assert validate_args.capability_action == "validate-contracts"
