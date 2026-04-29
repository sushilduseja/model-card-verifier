# Domain docs

## Layout

Single-context — one global `CONTEXT.md` at the repo root.

## Reading

- `CONTEXT.md` at repo root — domain language, terminology, project overview
- `docs/adr/` at repo root — architectural decision records (if any)

## Skills that read this

- `improve-codebase-architecture` — reads CONTEXT.md to understand domain
- `diagnose` — reads CONTEXT.md before root cause analysis
- `tdd` — reads CONTEXT.md for domain terminology

## Adding context

Create `CONTEXT.md` at repo root describing:
- What this project does
- Domain terminology
- Key architectural patterns

Example:

```markdown
# Project Context

This repo verifies model cards for AI models on HuggingFace.

## Domain Terms

- claim: an explicit capability or limitation stated in a model card
- test case: a prompt designed to verify or falsify a claim
- compliance score: percentage of claims the model satisfies
```