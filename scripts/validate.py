#!/usr/bin/env python3
"""Validate extensions.json against the JSON Schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    registry = json.loads((root / "extensions.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "schema" / "extensions.schema.json").read_text(encoding="utf-8"))

    try:
        import jsonschema
        jsonschema.validate(registry, schema)
        print(f"✓ extensions.json is valid ({len(registry['extensions'])} extension(s))")
    except ImportError:
        print("jsonschema not installed — doing basic checks only")
        assert "extensions" in registry, "Missing 'extensions' key"
        assert isinstance(registry["extensions"], list), "'extensions' must be a list"
        names = set()
        for ext in registry["extensions"]:
            assert "name" in ext, f"Extension missing 'name': {ext}"
            assert "pypi" in ext, f"Extension {ext['name']!r} missing 'pypi'"
            assert ext["pypi"].startswith("aar-ext-"), f"PyPI name must start with aar-ext-: {ext['pypi']}"
            assert ext["name"] not in names, f"Duplicate extension name: {ext['name']}"
            names.add(ext["name"])
        print(f"✓ Basic validation passed ({len(registry['extensions'])} extension(s))")
    except jsonschema.ValidationError as exc:
        print(f"✗ Validation failed: {exc.message}")
        return 1

    # Check for duplicate names
    names = [e["name"] for e in registry["extensions"]]
    dupes = [n for n in names if names.count(n) > 1]
    if dupes:
        print(f"✗ Duplicate extension names: {set(dupes)}")
        return 1

    # Check for duplicate pypi slugs
    slugs = [e["pypi"] for e in registry["extensions"]]
    dupe_slugs = [s for s in slugs if slugs.count(s) > 1]
    if dupe_slugs:
        print(f"✗ Duplicate PyPI slugs: {set(dupe_slugs)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
