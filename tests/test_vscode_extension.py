import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = REPO_ROOT / "tools" / "vscode-apg"


def _load_json(relative_path: str) -> dict:
    return json.loads((EXTENSION_ROOT / relative_path).read_text(encoding="utf-8"))


def test_package_json_valid():
    package = _load_json("package.json")

    assert package["name"] == "vscode-apg"
    assert "contributes" in package
    assert "engines" in package


def test_package_json_has_apg_language():
    package = _load_json("package.json")
    language = package["contributes"]["languages"][0]

    assert ".apg" in language["extensions"]


def test_grammar_valid():
    grammar = _load_json("syntaxes/apg.tmLanguage.json")

    assert grammar["scopeName"] == "source.apg"
    assert "patterns" in grammar


def test_snippets_valid():
    snippets = _load_json("snippets/apg.json")

    assert "module" in snippets
    assert "entity" in snippets
    assert "enum" in snippets


def test_language_config_valid():
    config = _load_json("language-configuration.json")

    assert "comments" in config
