# APG Utility Scripts

Repository automation lives under this directory so root-level files stay limited
to project entry points, metadata, and packaging configuration.

## Directories

- `capability_generation/` - one-off generators for composable capability
  definitions and capability metadata.
- `template_generation/` - setup and structure generators for APG application
  template assets.
- `migrations/` - migration utilities for platform structure transitions.

Run these scripts from the repository root unless a script explicitly documents a
different working directory. The moved scripts resolve the repository root from
their own file location before importing APG modules or writing template assets.
