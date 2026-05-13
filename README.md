# Aar Extensions Registry

Curated registry and first-party extension packages for the [Aar](https://github.com/fischerf/aar) agent framework.

## Structure

    extensions.json              - the registry (source of truth)
    schema/                      - JSON Schema for validation
    scripts/
      validate.py                - validate extensions.json
      build_site.py              - generate static HTML site
    site/                        - generated HTML (deploy to GitHub Pages)
    packages/                    - first-party seed extension packages
      aar-ext-permission-gate/   - block dangerous bash commands
      aar-ext-protected-paths/   - block writes to sensitive files
      aar-ext-git-checkpoint/    - auto-commit at turn boundaries
      aar-ext-mcp-tools/         - MCP server tool discovery via extension API
      aar-ext-observability/     - structured metrics per turn

## First-Party Extensions

| Package | Description | Status |
|---|---|---|
| [aar-ext-permission-gate](packages/aar-ext-permission-gate/) | Block dangerous bash commands (rm -rf, sudo, mkfs, etc.) | Ready |
| [aar-ext-protected-paths](packages/aar-ext-protected-paths/) | Block writes to .env, secrets, credentials, SSH keys | Ready |
| [aar-ext-git-checkpoint](packages/aar-ext-git-checkpoint/) | Auto-commit at turn boundaries + rollback tool | Ready |
| [aar-ext-mcp-tools](packages/aar-ext-mcp-tools/) | MCP server tool discovery via extension API | Reference |
| [aar-ext-observability](packages/aar-ext-observability/) | Structured metrics/logging per turn | Ready |

Each package has its own pyproject.toml, source, README, and tests.

## Adding an Extension

1. Fork this repo
2. Add your extension to extensions.json
3. (Optional) Add the package under packages/ if first-party
4. Run python scripts/validate.py to check
5. Open a pull request

## Running Tests

Run tests for a single package:

    cd packages/aar-ext-permission-gate
    python -m pytest tests/ -v

## Building the Site

    python scripts/build_site.py

Deploy site/ to GitHub Pages or any static host.

## Validation

    pip install jsonschema
    python scripts/validate.py

## For Extension Authors

See the [Aar Extension Developer Guide](https://github.com/fischerf/aar/blob/main/docs/extensions.md).

## Author

**Florian Fischer** — [Discord](https://discord.gg/xYJNHJV7Bh)

## License

Apache License 2.0 same as Aar
