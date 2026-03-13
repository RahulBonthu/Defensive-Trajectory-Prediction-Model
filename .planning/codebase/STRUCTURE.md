# Codebase Structure

**Analysis Date:** 2026-03-13

## Current Directory Layout

```
Defensive Trajectory Prediction Model/
├── .claude/                    # GSD framework configuration
│   ├── agents/                 # Claude agent definitions
│   ├── commands/               # GSD command implementations
│   ├── hooks/                  # Git and session hooks
│   ├── get-shit-done/          # Framework core
│   ├── settings.json           # GSD settings
│   └── gsd-file-manifest.json  # File integrity tracking
├── .git/                       # Git repository
├── .planning/                  # Planning documents
│   └── codebase/               # Codebase analysis (this location)
└── (no source files yet)
```

## Key Directories to Create

**Project Root Structure (recommended layout):**

```
src/                           # Primary source code
├── data/                       # Data processing and utilities
├── models/                     # Machine learning models
├── services/                   # Service/API layer
├── utils/                      # Shared utilities
└── __init__.py                 # Package initialization

tests/                          # Test suite
├── unit/                       # Unit tests
├── integration/                # Integration tests
└── fixtures/                   # Test data and fixtures

notebooks/                      # Jupyter notebooks (optional)
├── exploration/                # Data exploration
└── analysis/                   # Analysis and visualization

data/                           # Data directory (development)
├── raw/                        # Raw input data
├── processed/                  # Processed data
└── models/                     # Trained model artifacts

config/                         # Configuration files
├── default.yaml                # Default configuration
└── training.yaml               # Training parameters

docs/                           # Documentation
├── README.md                   # Project overview
├── SETUP.md                    # Setup instructions
└── API.md                      # API documentation (if applicable)

scripts/                        # Utility scripts
├── train.py                    # Training script
└── predict.py                  # Prediction/inference script
```

## Recommended File Organization

**Python Package Structure:**

- **Entry points:** `src/` directory
- **Tests:** `tests/` directory (parallel structure to src/)
- **Configuration:** `config/` directory
- **Data:** `data/` directory (for development/testing)
- **Documentation:** Project root level

## Directory Purposes (To Be Implemented)

**src/ - Source Code:**
- Purpose: Contains all production code
- Contains: Python modules for data processing, ML models, and services
- Key files: `__init__.py` (package markers), core modules

**tests/ - Test Suite:**
- Purpose: Automated testing
- Contains: Unit tests, integration tests, test fixtures
- Key files: `conftest.py` (pytest configuration), `test_*.py` files

**config/ - Configuration:**
- Purpose: Application configuration management
- Contains: YAML/JSON configuration files
- Key files: `default.yaml`, environment-specific configs

**data/ - Data Directory:**
- Purpose: Store training/test data and models
- Contains: Raw data, processed data, trained models
- Key files: Data files (CSV, JSON, etc.), model artifacts

**docs/ - Documentation:**
- Purpose: Project documentation
- Contains: README, API docs, setup guides
- Key files: Markdown documentation files

## Naming Conventions (To Follow)

**Files:**
- `module_name.py` - Snake case for Python files
- `test_module_name.py` - Test files prefixed with `test_`
- `conftest.py` - Pytest configuration
- `README.md` - Markdown documentation

**Directories:**
- `lowercase_with_underscores` - Snake case for directories
- `src/`, `tests/`, `config/`, `data/`, `docs/` - Standard names

**Python Code (to be implemented):**
- Classes: `PascalCase` (e.g., `TrajectoryPredictor`)
- Functions: `snake_case` (e.g., `predict_trajectory`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- Private members: `_leading_underscore`

## Special Directories

**.claude/ - GSD Framework:**
- Purpose: Get Shit Done automation system
- Generated: Partially (framework files)
- Committed: Yes (framework required for operations)
- Do not modify framework files without understanding GSD system

**.git/ - Version Control:**
- Purpose: Git repository metadata
- Generated: Yes (by git init)
- Committed: No (internal git data)

**.planning/codebase/ - Analysis Documents:**
- Purpose: Codebase analysis and architecture documentation
- Generated: Yes (by gsd:map-codebase)
- Committed: Yes
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md

## Where to Add New Code

**For new features/modules:**
- Implementation code: `src/[feature_module]/`
- Tests: `tests/[feature_module]/test_*.py`
- Follow snake_case naming for both files and functions

**For data processing utilities:**
- Location: `src/data/` directory
- Example: `src/data/preprocessor.py`, `src/data/loaders.py`

**For ML model code:**
- Location: `src/models/` directory
- Example: `src/models/trajectory_predictor.py`, `src/models/loss_functions.py`

**For API/service endpoints:**
- Location: `src/services/` directory
- Example: `src/services/prediction_api.py`

**For shared utilities:**
- Location: `src/utils/` directory
- Example: `src/utils/validators.py`, `src/utils/config.py`

**For scripts:**
- Location: `scripts/` directory
- Example: `scripts/train.py`, `scripts/evaluate.py`

---

*Structure analysis: 2026-03-13 - Project initialization stage*
