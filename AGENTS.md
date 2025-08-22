# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `quant_math_lab.py` (workbench + self-tests), `qcqi_pure_math_playground.py` (utilities + demos).
- Notebooks: `*.ipynb` for exploration; not required to run scripts.
- Assets: reference PDFs. No packaging yet; minimal `requirements.txt` (NumPy only).

## Build, Test, and Development Commands
- Create env + install deps:
  - `python -m venv .venv && source .venv/bin/activate` (or `Scripts\\activate` on Windows)
  - `pip install -r requirements.txt`
- Run self-tests and demos:
  - `python quant_math_lab.py` (prints “✅ Alle zelftests geslaagd.” on success)
  - `python qcqi_pure_math_playground.py` (prints CHSH/entropy/channel examples)

## Coding Style & Naming Conventions
- Python, 4-space indent, NumPy-only (avoid extra deps).
- Functions/variables: `snake_case`. Constants/gates: `UPPERCASE` (e.g., `X`, `Z`, `H`, `I2`).
- Type hints where practical (`numpy.typing.NDArray`), concise docstrings.
- Indexing convention: tensor factors are MSB→LSB (left→right). `partial_trace` follows this.
- Keep helpers pure, side-effect free; prefer small, composable functions.

## Testing Guidelines
- Primary: extend `run_self_tests()` in `quant_math_lab.py` and the `__main__` checks in `qcqi_pure_math_playground.py`.
- Add quick numerical assertions (e.g., `np.allclose(..., atol=1e-9)`).
- If introducing larger features, add a lightweight test script under `tests/` (optional), but do not add heavy frameworks unless agreed.

## Commit & Pull Request Guidelines
- Commits: small, focused; recommended prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `tests:`.
- PRs must include:
  - Summary of changes and motivation.
  - Repro steps and sample output (paste self-test lines).
  - Any API or convention changes (update README/this file accordingly).
- Do not break demos or indexing conventions; preserve NumPy-only scope.

## Tips & Conventions
- Angles are radians; dimensions as lists (e.g., `[2,3,2]`).
- Keep memory usage reasonable; prefer `einsum`/reshapes to Python loops where possible.
- When adding channels/algos, include a minimal usage snippet in the module’s `__main__`.

