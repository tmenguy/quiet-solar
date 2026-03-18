# CLAUDE.md

Quiet Solar is a Home Assistant custom component that optimizes solar energy self-consumption through a constraint-based solver.

## Commands

ALWAYS use the project's virtual environment (`./venv`) for running pytest and all Python commands.

```bash
# Activate venv first
source venv/bin/activate

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_solver.py

# Run a single test
pytest tests/test_solver.py::test_function_name -v

# Run with coverage (100% is mandatory)
pytest tests/ --cov=custom_components/quiet_solar --cov-report=term-missing

# Run only unit or integration tests
pytest tests/ -m unit
pytest tests/ -m integration
```

## Full Documentation

Before implementing any code, read these documents:

- **Code-level rules (42 rules)**: `_bmad-output/project-context.md` — naming, async, logging, error handling, testing anti-patterns
- **Architecture & patterns (9 patterns)**: `_bmad-output/planning-artifacts/architecture.md` — component model, decision map, implementation patterns, project structure, CI/CD strategy
