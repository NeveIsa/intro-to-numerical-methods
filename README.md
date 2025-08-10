# Intro to Numerical Methods

A collection of interactive Python notebooks (built with [Marimo]) exploring core numerical methods: root finding, polynomial deflation, gradient descent, and linear transformations. Notebooks are plain Python files that you can run or edit in a browser UI.

## Getting Started

- Python: 3.10+ recommended
- Install dependencies from `req.txt` (Marimo, NumPy, Seaborn, tqdm)

Setup

1) Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

2) Install requirements

```
pip install -r req.txt
```

## Running the Notebooks

You can run or edit any notebook with Marimo.

- Run mode (executes cells, simple UI):

```
marimo run notebooks/1-root-finding.py
```

- Edit mode (full interactive editor in browser):

```
marimo edit notebooks/10-gradient-descent.py
```

Common entries:
- `notebooks/1-root-finding.py` – classic root-finding methods and visuals
- `notebooks/2-deflation.py` – polynomial deflation workflows
- `notebooks/10-gradient-descent.py` – 1D gradient descent demo
- `notebooks/11-gradient-descent-2D.py` – 2D gradient descent demo

Additional materials live under `sumedh-notebooks/` and `numerical/` (utility functions).

## Repository Layout

- `notebooks/`: main interactive lessons and demos
- `numerical/`: helper functions for numeric routines
- `sumedh-notebooks/`: supplemental notebooks and examples
- `EXTRA/`, `paper/`, `server/`: project-specific assets (optional)
- `req.txt`: Python dependencies for quick setup

## Development Notes

- Cache files (`__pycache__/`, `*.pyc`) are ignored via `.gitignore`.
- If you pull changes that create merge conflicts in notebooks, prefer resolving by keeping whole-file versions (these are generated notebooks). When in doubt, pick the version that matches your local Marimo version.

## Troubleshooting

- Marimo not found: ensure the virtual environment is active and `marimo` is installed.
- Browser doesn’t open: copy the local URL Marimo prints and open it manually.
- Plotting issues: ensure `seaborn` and `matplotlib` are installed from `req.txt`.

## License

Educational use. If you need a specific license, add one here.

[Marimo]: https://marimo.io
