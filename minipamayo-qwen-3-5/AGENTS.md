# AGENTS.md

This file defines repository-specific implementation rules for `minipamayo-qwen-3-5`.

## Canonical Artifacts

- Treat config JSON, dataset records, extract summaries, checkpoint metadata, and run metadata as canonical schemas.
- Do not add backward-compatibility code unless explicitly requested.
- Do not silently recover from malformed or incomplete artifacts.

## No Silent Fallbacks

- Do not use `.get(..., default)` for required pipeline fields.
- Do not use `x or default` to fill required values.
- Do not guess alternate file paths for required artifacts.
- Do not return `None` from `require_*` helpers.
- If a required field, path, or artifact is missing, raise `RuntimeError` immediately with the concrete missing key or path.

## Boundary Validation

- Validate raw dictionaries at the boundary, then pass normalized data deeper into the pipeline.
- Prefer `require_*` helpers or typed schema objects over ad hoc field access.
- In `src/minipamayo_qwen35/data`, `src/minipamayo_qwen35/train`, `src/minipamayo_qwen35/eval`, and related utils, required schema fields must be accessed strictly.

## Change Policy

- When tightening schemas, prefer making failures explicit over preserving old broken artifacts.
- If compatibility is needed later, add it only when explicitly requested and isolate it behind clearly named migration code.
