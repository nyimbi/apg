# APG Language Support

VS Code language support for APG DSL files.

## Features

- Syntax highlighting for `.apg` modules, entities, tables, enums, security blocks, relationships, field types, validators, strings, numbers, comments, and punctuation.
- Snippets for modules, entities, enums, relationships, security blocks, and validated string fields.
- Language configuration for line comments, block comments, brackets, auto-closing pairs, and APG identifiers.

## Installation

From this directory, package the extension:

```sh
npx @vscode/vsce package
```

Install the generated VSIX:

```sh
code --install-extension vscode-apg-0.1.0.vsix
```

For development, open this folder in VS Code and press `F5` to launch an Extension Development Host.

## Example

```apg
module customer_records {

  entity Customer {
    id: uuid @required;
    name: str @min_length(1) @max_length(120);
    email: str @email @optional;
  }

  enum CustomerStatus {
    active;
    inactive;
  }

  security {
    authentication: required;
  }
}
```
