# Repository Guidelines

## Project Structure & Module Organization

This repository develops entity resolution for the AMLC-2026 business matching task. `blocking/` contains the Python candidate-generation pipeline, its evaluation and smoke-test scripts, requirements, and recorded results. `eda/` contains exploratory analysis scripts, the notebook, findings, and generated figures. `diagram/` stores pipeline diagrams and editable Mermaid/Excalidraw sources. `subagents/` contains research notes. Project constraints and measured dataset findings live in `PROJECT_CONTEXT.md` and `PROJECT_OVERVIEW.md`.

## Build, Test, and Development Commands

Install blocker dependencies with `python -m pip install -r blocking/requirements.txt`. From the repository root:

- `PYTHONPATH=. python blocking/smoke_test.py india 2000 60000 60000` runs the documented end-to-end sanity check.
- `PYTHONPATH=. python blocking/measure_recall_at_k.py --sample 3000 --s2s3-cap 80000 --k 10,20,50,100,200` measures candidate recall across K values.
- `PYTHONPATH=. python -m blocking.blocker_v1 --dry-run` exercises the smallest partition without writing candidate output.

The Mermaid CLI is managed separately in `diagram/mermaid/`; see its README for rendering instructions. No repository-wide build or formatter is configured.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for modules/functions/variables, and `CapWords` for classes. Keep configuration in `blocking/config.py` and reuse the normalization and streaming I/O helpers. Preserve deterministic output and country partitioning. Unicode-safe punctuation handling is important for Devanagari; do not replace it with ASCII-centric cleanup.

## Testing Guidelines

There is no configured pytest suite or stated coverage threshold. Use `blocking/smoke_test.py` for end-to-end changes and the recall evaluator when changing candidate generation or ranking. Keep ground truth confined to evaluation code; candidate generation must not read it. Record meaningful measurement commands and outputs under `blocking/results/` when adding experiment evidence.

## Commit & Pull Request Guidelines

Git history currently contains only the initial commit, so no established message convention can be inferred. Use a short imperative subject that names the change, such as `Preserve Unicode marks during normalization`. Pull requests should explain the motivation and behavior change, list validation commands and measured results, link related issues when available, and include before/after images for diagram or notebook visual changes.

## Data & Competition Constraints

Keep raw datasets and generated large candidate files out of version control. Do not use external APIs, databases, or services to identify businesses or enrich records; this violates the competition constraints documented in `PROJECT_CONTEXT.md`.
