# Python Package Setup & Import Guide

This project is organized as a standard Python package to ensure robust, absolute imports across different subdirectories (like `src/` and `src/ml_watcher/`).

## 1. Project Structure

The project follows a modular structure where the `src` folder acts as the root package:

```text
.
├── setup.py           # Package configuration
├── src/               # Root package folder
│   ├── __init__.py    # Marks 'src' as a package
│   ├── monitor.py     # Central monitoring logic
│   └── watcher/       # Watcher subsystem
│       ├── __init__.py
│       └── ml_watcher/  # Machine Learning sub-package
│           ├── __init__.py
│           ├── watcher_xgboost.py
│           └── verify_watcher.py
└── .venv/             # Virtual environment
```

## 2. Setup (Mandatory)

To enable absolute imports (e.g., `from src.monitor import ...`), you must install the project in **editable mode**. This tells Python that the `src` directory is a library.

Run this command from the project root:

```bash
pip install -e .
```

> [!NOTE]
> The `-e` flag (Editable) means you only need to run this **once**. If you move or rename files later, Python will automatically track the changes without requiring a reinstall.

## 3. How to Import

Now you can import modules from anywhere in the codebase using their full path starting with `src`:

```python
# In any file inside or outside src/watcher/ml_watcher/
from src.monitor import DetectionAgent
from src.watcher.ml_watcher.watcher_xgboost import XGBoostWatcher
```

## 4. Running Scripts

Since the project is a package, it is recommended to run scripts using your virtual environment from the root directory:

```bash
# Recommended way
python3 src/ml_watcher/verify_watcher.py

# Alternative (as a module)
python3 -m src.ml_watcher.verify_watcher
```

## 5. Troubleshooting

If you see `ModuleNotFoundError: No module named 'src'`, verify that:
1. You have activated your virtual environment.
2. You have run `pip install -e .` in that environment.
3. There is an `__init__.py` file in both `src/` and `src/ml_watcher/`.
