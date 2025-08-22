# Quantum Math Lab + QCQI Pure Math Playground

Two small, NumPy‑only modules to explore quantum mechanics and quantum information math. They include linear‑algebra helpers, gates, density matrices, partial trace, CHSH, simple oracles (Deutsch/Deutsch–Jozsa), QFT/QPE simulation, Bloch vectors, entropies, fidelity/trace distance, and simple channels (Kraus/Choi).

Most docstrings and comments are in Dutch; function names are concise and conventional. No external frameworks are required beyond NumPy.

## Requirements
- Python 3.9+
- NumPy (install via requirements file)

Install:
```
pip install -r requirements.txt
```

## Contents
- `quant_math_lab.py`: compact “workbench” with helpers and small algorithms.
  - Linear algebra: `ket`, `bra`, `dagger`, `tensor`, `projector`, `dm`, `is_unitary`.
  - Operators: `X`, `Y`, `Z`, `H`, rotations `Rx/Ry/Rz`, `CNOT`, `SWAP`, `controlled(U)`.
  - Density matrices: Born probabilities, partial trace, expectations, variances, commutators, uncertainty bound.
  - Bell/CHSH, Deutsch and Deutsch–Jozsa, QFT, QPE simulation.
  - Self‑tests via `run_self_tests()` under `__main__`.

- `qcqi_pure_math_playground.py`: focused utilities and demos.
  - Kets, Pauli, Bloch vector extraction, Bell states and CHSH.
  - Entropies (von Neumann, Rényi‑2), fidelity, trace distance.
  - Channels: dephasing via Kraus, Choi construction, CPTP checks.
  - Robust `partial_trace` supporting arbitrary local dimensions and keep‑sets.

## Usage

Run self‑tests and examples:
```
python quant_math_lab.py
python qcqi_pure_math_playground.py
```

Expected outputs include (indicative):
- CHSH for `|phi+>` ≈ `2*sqrt(2)`.
- QPE demo returns a bitstring `y` with `y/2^m` close to the true phase.
- Dephasing drives the Bloch vector toward the Z‑axis (off‑diagonals shrink).

## Notes
- Indexing convention for tensor products is MSB→LSB (left→right): subsystem 0 is the leftmost factor in `tensor(A,B,...)`/`kron(A,B,...)`. The `partial_trace` helpers follow this convention.
- The repository also contains a few Jupyter notebooks and PDFs as references; they’re not required to run the scripts.

## Troubleshooting
- If NumPy is missing or too old, reinstall via `pip install -r requirements.txt`.
- For complex number display issues in some terminals, ensure `repr`/printing isn’t truncated by your environment.

