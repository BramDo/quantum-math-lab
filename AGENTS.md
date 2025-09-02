# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `quant_math_lab.py` (workbench + self-tests) and `qcqi_pure_math_playground.py` (utilities + demos).
- Notebooks: `*.ipynb` for exploration only; scripts do not depend on them.
- Assets: reference PDFs (e.g., in `assets/`). Minimal `requirements.txt` (NumPy only).
- Optional lightweight tests may live under `tests/` when helpful.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate` (Windows: `Scripts\activate`).
- Install deps: `pip install -r requirements.txt`.
- Run self-tests: `python quant_math_lab.py` → expects “✅ Alle zelftests geslaagd.” on success.
- Run demos: `python qcqi_pure_math_playground.py` → prints CHSH/entropy/channel examples.

## Coding Style & Naming Conventions
- Language: Python, 4‑space indent, NumPy‑only (avoid extra deps).
- Names: `snake_case` for functions/variables; UPPERCASE for constants/gates (`X`, `Z`, `H`, `I2`).
- Types/Docs: add type hints (e.g., `numpy.typing.NDArray`) and concise docstrings.
- Indexing: tensor factors are MSB→LSB (left→right). `partial_trace` follows this convention.
- Design: keep helpers pure and small; prefer `einsum`/reshapes over Python loops.

## Testing Guidelines
- Primary path: extend `run_self_tests()` in `quant_math_lab.py` and `__main__` checks in `qcqi_pure_math_playground.py`.
- Use quick numerics: `np.allclose(..., atol=1e-9)` for assertions.
- Larger features: add a tiny script under `tests/` if needed; do not add heavy frameworks.
- Run: `python quant_math_lab.py` as the default verification step.

## Commit & Pull Request Guidelines
- Commits: small and focused; recommended prefixes `feat:`, `fix:`, `refactor:`, `docs:`, `tests:`.
- PRs must include: summary/motivation, repro steps with sample output (paste key self‑test lines), and any API or convention changes (update this file/README where relevant).
- Constraints: do not break demos or indexing conventions; preserve NumPy‑only scope.

## Tips & Conventions
- Angles are radians; dimensions as lists (e.g., `[2,3,2]`).
- Keep memory usage reasonable; avoid unnecessary copies; prefer vectorized NumPy.
- When adding channels/algos, include a minimal usage snippet in the module’s `__main__`.

