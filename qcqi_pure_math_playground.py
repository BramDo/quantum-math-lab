"""
QCQI Pure Math Playground (zonder Qiskit)

Gebruik: kopieer dit script naar je eigen omgeving en run het met Python.
Bevat helpers voor states/dichtheidsmatrices, Bloch-vectoren, Bell/CHSH,
entropie (von Neumann & Renyi-2), fidelity/trace distance, en Kraus/Choi-kanalen.

Let op (indexeringsconventie):
- In `partial_trace(rho, keep, dims)` verwijzen subsystem-indexen 0..n-1 naar de
  linker-naar-rechter volgorde waarin je met `kron(A, B, C, ...)` hebt opgebouwd.
  Dus index 0 = meest-linkse factor (MSB), index n-1 = rechterste (LSB).
  Als je liever 0=LSB hanteert, laat het weten; dan pas ik de helpers daarop aan.
"""

import numpy as np
from functools import reduce

# =============================
# Basis: kets en Pauli-matrices
# =============================
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

# ======================
# Lineaire-algebra utils
# ======================

def kron(*ops):
    """Kronecker product links->rechts."""
    return reduce(np.kron, ops)


def normalize(state):
    nrm = np.linalg.norm(state)
    return state if nrm == 0 else state / nrm


def to_density(state):
    """|psi><psi| voor een pure toestand."""
    psi = normalize(state)
    return np.outer(psi, np.conjugate(psi))


def basis_state(bitstring: str) -> np.ndarray:
    """Maak |bitstring> (bijv. '010')."""
    n = len(bitstring)
    idx = int(bitstring, 2)
    vec = np.zeros(2**n, dtype=complex)
    vec[idx] = 1.0
    return vec


def probs_z(state, n_qubits):
    """Meetkansen in Z-basis voor pure statevector."""
    psi = normalize(state)
    p = np.abs(psi) ** 2
    return {format(i, f"0{n_qubits}b"): float(p[i]) for i in range(len(psi))}


def partial_trace(rho: np.ndarray, keep, dims=None) -> np.ndarray:
    """Partiele trace over alle subsystemen behalve die in `keep`.

    Conventie: subsystem-indexen 0..n-1 volgen de orde in `kron(A, B, C, ...)`.
    0 is de meest-linkse factor (MSB).

    Parameters
    ----------
    rho : np.ndarray
        Dichtheidsmatrix met vorm (D, D), D = prod(dims).
    keep : iterable[int]
        Subsystemen (indices) die behouden blijven. De volgorde bepaalt de
        volgorde van subsystemen in de output.
    dims : list[int] | None
        Lokale dimensies per subsysteem. Default: allemaal 2.

    Returns
    -------
    np.ndarray
        Gereduceerde dichtheidsmatrix met vorm (prod(d_keep), prod(d_keep)).

    Implementatie-opmerking
    -----------------------
    In plaats van herhaaldelijk te traceren met veranderende as-indices (wat
    foutgevoelig is), permuteren we de tensor-assen naar de vorm
    (keep_row, throw_row, keep_col, throw_col) en gebruiken we vervolgens
    `einsum('atbt->ab', ...)` om over de weggegooide dimensies te traceren.
    Hiermee voorkomen we out-of-bounds as-selecties.
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho moet een vierkante matrix zijn.")

    if dims is None:
        n_guess = int(round(np.log2(rho.shape[0])))
        if 2**n_guess != rho.shape[0]:
            raise ValueError("dims niet opgegeven en dim is geen macht van 2.")
        dims = [2] * n_guess

    n = len(dims)
    if int(np.prod(dims)) != rho.shape[0]:
        raise ValueError("Product van dims komt niet overeen met rho-dimensie.")

    keep = tuple(int(k) for k in keep)
    throw = tuple(i for i in range(n) if i not in keep)

    # Reshape naar rang-2n tensor (d0,..,d_{n-1}, d0,..,d_{n-1})
    tens = rho.reshape(*dims, *dims)

    # Permuteer naar (keep_row, throw_row, keep_col, throw_col)
    perm = list(keep) + list(throw) + [n + i for i in keep] + [n + i for i in throw]
    tens = tens.transpose(perm)

    # Her-reshape naar (Dk, Dt, Dk, Dt)
    Dk = int(np.prod([dims[i] for i in keep])) if len(keep) > 0 else 1
    Dt = int(np.prod([dims[i] for i in throw])) if len(throw) > 0 else 1
    tens = tens.reshape(Dk, Dt, Dk, Dt)

    # Trace over de weggegooide subsystemen: som over t (diag) -> einsum
    # Resultaat: (Dk, Dk)
    reduced = np.einsum('atbt->ab', tens)
    return reduced

# ==================
# Bloch-vector (1q)
# ==================

def bloch_vector_from_rho(rho1: np.ndarray) -> np.ndarray:
    """Geef Bloch-vector (<X>,<Y>,<Z>) terug voor 1-qubit rho."""
    return np.array([
        np.real(np.trace(rho1 @ X)),
        np.real(np.trace(rho1 @ Y)),
        np.real(np.trace(rho1 @ Z)),
    ])

# =========================
# Bell-staten & CHSH-waarde
# =========================

def bell_state(which: str = "phi+") -> np.ndarray:
    if which == "phi+":
        return normalize((kron(ket0, ket0) + kron(ket1, ket1)) / np.sqrt(2))
    if which == "phi-":
        return normalize((kron(ket0, ket0) - kron(ket1, ket1)) / np.sqrt(2))
    if which == "psi+":
        return normalize((kron(ket0, ket1) + kron(ket1, ket0)) / np.sqrt(2))
    if which == "psi-":
        return normalize((kron(ket0, ket1) - kron(ket1, ket0)) / np.sqrt(2))
    raise ValueError("kies uit {'phi+','phi-','psi+','psi-'}")


def obs_equator(theta: float) -> np.ndarray:
    """Observable in X-Z vlak: cos(theta) Z + sin(theta) X."""
    return np.cos(theta) * Z + np.sin(theta) * X


def chsh_value(state: np.ndarray, a0: float, a1: float, b0: float, b1: float) -> float:
    A0, A1 = obs_equator(a0), obs_equator(a1)
    B0, B1 = obs_equator(b0), obs_equator(b1)
    CHSH = kron(A0, B0 + B1) + kron(A1, B0 - B1)
    rho = to_density(state)
    return float(np.real(np.trace(CHSH @ rho)))

# ===============================
# Entropie (vN, Renyi-2) & hulpjes
# ===============================

def von_neumann_entropy(rho: np.ndarray, base=np.e) -> float:
    vals = np.clip(np.linalg.eigvalsh(rho), 0.0, 1.0)
    nz = vals[vals > 1e-12]
    return float(-np.sum(nz * (np.log(nz) / np.log(base))))


def renyi2_entropy(rho: np.ndarray, base=np.e) -> float:
    return float(-np.log(np.real(np.trace(rho @ rho))) / np.log(base))


def entanglement_entropy_pure(psi: np.ndarray, A_qubits, n_qubits: int, base=np.e):
    rho = to_density(psi)
    rhoA = partial_trace(rho, keep=tuple(A_qubits), dims=[2] * n_qubits)
    return von_neumann_entropy(rhoA, base=base), renyi2_entropy(rhoA, base=base)

# ======================================
# Vergelijkingsmaten: fidelity & T-distance
# ======================================

def matrix_sqrt(rho: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0.0, None)
    return (vecs @ np.diag(np.sqrt(vals)) @ np.conjugate(vecs).T).astype(complex)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(rho,sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    rs = matrix_sqrt(rho) @ sigma @ matrix_sqrt(rho)
    vals = np.linalg.eigvalsh(matrix_sqrt(rs))
    return float(np.real(np.sum(vals)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """T = 0.5 * ||rho - sigma||_1 (via SVD)."""
    diff = rho - sigma
    svals = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * float(np.sum(np.abs(svals)))

# ======================================
# Kanalen: Kraus-representatie & Choi
# ======================================

def choi_from_kraus(kraus_ops):
    r"""Choi J(E) = sum_i (K_i \otimes I) |Phi><Phi| (K_i^\dagger \otimes I), |Phi>=sum_j |jj>/\sqrt(d)."""
    d = kraus_ops[0].shape[0]
    phi = np.zeros((d * d, 1), dtype=complex)
    for j in range(d):
        e = np.zeros((d, 1), dtype=complex)
        e[j, 0] = 1
        phi += np.kron(e, e)
    phi = phi / np.sqrt(d)
    choi = np.zeros((d * d, d * d), dtype=complex)
    for K in kraus_ops:
        choi += np.kron(K, np.eye(d)) @ (phi @ np.conjugate(phi).T) @ np.kron(np.conjugate(K.T), np.eye(d))
    return choi


def is_cptp(kraus_ops, tol=1e-9):
    """Retourneer (OK, TP_ok, CP_ok, Choi-eigenwaarden)."""
    acc = np.zeros((2, 2), dtype=complex)
    for K in kraus_ops:
        acc += np.conjugate(K.T) @ K
    tp_ok = np.allclose(acc, np.eye(2), atol=tol)
    evals = np.linalg.eigvalsh(choi_from_kraus(kraus_ops))
    cp_ok = np.all(evals >= -1e-9)
    return tp_ok and cp_ok, tp_ok, cp_ok, evals


def dephasing_kraus(p: float):
    K0 = np.sqrt(1 - p) * np.eye(2)
    K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
    return [K0, K1]


def dephase_rho(rho: np.ndarray, p: float) -> np.ndarray:
    Ks = dephasing_kraus(p)
    out = np.zeros_like(rho, dtype=complex)
    for K in Ks:
        out += K @ rho @ np.conjugate(K.T)
    return out


def amplitude_damping_kraus(gamma: float):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


def amplitude_damp_rho(rho: np.ndarray, gamma: float) -> np.ndarray:
    Ks = amplitude_damping_kraus(gamma)
    out = np.zeros_like(rho, dtype=complex)
    for K in Ks:
        out += K @ rho @ np.conjugate(K.T)
    return out

# ========
# Playground & Tests
# ========
if __name__ == "__main__":
    # 1) Bloch-vector van |+>
    rho_plus = to_density((ket0 + ket1) / np.sqrt(2))
    print("Bloch <X,Y,Z> voor |+>:", bloch_vector_from_rho(rho_plus))

    # 2) CHSH op |phi+>
    psi = bell_state("phi+")
    a0, a1 = 0.0, np.pi / 2
    b0, b1 = np.pi / 4, -np.pi / 4
    print("CHSH(|phi+>):", chsh_value(psi, a0, a1, b0, b1))

    # 3) Entanglement-entropie van GHZ (A vs BC)
    ghz = normalize((kron(ket0, ket0, ket0) + kron(ket1, ket1, ket1)) / np.sqrt(2))
    SvN, S2 = entanglement_entropy_pure(ghz, A_qubits=(0,), n_qubits=3)
    print("GHZ entanglement: SvN=", SvN, " S2=", S2)

    # 4) Fidelity en trace distance voorbeeld
    rho0 = to_density(kron(ket0, ket0))
    rho1 = to_density(kron(ket1, ket1))
    print("Fidelity(|00>,|11>) =", fidelity(rho0, rho1))
    print("Trace distance(|00>,|11>) =", trace_distance(rho0, rho1))

    # 5) Dephasing-kanaal CPTP-check
    ok, tp, cp, evals = is_cptp(dephasing_kraus(0.3))
    print("Dephasing CPTP:", ok, " (TP:", tp, ", CP:", cp, ") eigenwaarden Choi:", evals)

    # 6) Dephasing-effect op |+>
    for p in (0.0, 0.3, 0.7, 1.0):
        rout = dephase_rho(rho_plus, p)
        vec = bloch_vector_from_rho(rout)
        print(f"p={p}  Bloch: {vec}")

    # 7) EXTRA TESTS: partial_trace basischecks (bestonden al)
    def _assert_close(A, B, name="test", tol=1e-9):
        ok = np.allclose(A, B, atol=tol)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print("  max|A-B|=", np.max(np.abs(A-B)))

    # 7a) Productstaat |0>|1> op 2 qubits
    psi01 = kron(ket0, ket1)
    rho01 = to_density(psi01)
    rho_left = partial_trace(rho01, keep=(0,), dims=[2,2])   # linker (MSB)
    rho_right = partial_trace(rho01, keep=(1,), dims=[2,2])  # rechter (LSB)
    _assert_close(rho_left, to_density(ket0), "PT keep left qubit == |0><0|")
    _assert_close(rho_right, to_density(ket1), "PT keep right qubit == |1><1|")

    # 7b) Bell |phi+>: gereduceerd is I/2
    bell = bell_state("phi+")
    rho_bell = to_density(bell)
    mix = 0.5 * I
    _assert_close(partial_trace(rho_bell, keep=(0,), dims=[2,2]), mix, "Bell reduce left = I/2")
    _assert_close(partial_trace(rho_bell, keep=(1,), dims=[2,2]), mix, "Bell reduce right = I/2")

    # 7c) Vormcheck met niet-uniforme dims (2 x 3 systeem)
    rng = np.random.default_rng(1234)
    v = rng.normal(size=6) + 1j*rng.normal(size=6)
    v = v / np.linalg.norm(v)
    rho23 = to_density(v)
    A = partial_trace(rho23, keep=(0,), dims=[2,3])  # 2x2
    B = partial_trace(rho23, keep=(1,), dims=[2,3])  # 3x3
    print("Niet-uniforme dims: shapes", A.shape, B.shape)

    # 7d) Nieuw: keep=() moet scalar (1x1) met Tr(rho)=1 opleveren
    r = partial_trace(rho23, keep=(), dims=[2,3])
    _assert_close(r, np.array([[1.0]], dtype=complex), "PT keep=() gives [[1]]")

    # 7e) Nieuw: random 3-qubit (2,2,2), vergelijk met langzame referentie
    def partial_trace_slow(rho_full, keep_idx, dims_loc):
        # brute force via uit-sommen over throw indices
        keep_idx = list(keep_idx)
        n = len(dims_loc)
        throw_idx = [i for i in range(n) if i not in keep_idx]
        # volgorde: keep dan throw
        dims_keep = [dims_loc[i] for i in keep_idx]
        dims_throw = [dims_loc[i] for i in throw_idx]
        Dk = int(np.prod(dims_keep)) if dims_keep else 1
        Dt = int(np.prod(dims_throw)) if dims_throw else 1
        tens = rho_full.reshape(*dims_loc, *dims_loc)
        perm = keep_idx + throw_idx + [n+i for i in keep_idx] + [n+i for i in throw_idx]
        tens = tens.transpose(perm).reshape(Dk, Dt, Dk, Dt)
        out = np.zeros((Dk, Dk), dtype=complex)
        for t in range(Dt):
            out += tens[:, t, :, t]
        return out

    dims_222 = [2,2,2]
    v3 = rng.normal(size=8) + 1j*rng.normal(size=8)
    v3 /= np.linalg.norm(v3)
    rho3 = to_density(v3)
    for keep_set in [(0,), (1,), (2,), (0,1), (1,2), (0,2)]:
        fast = partial_trace(rho3, keep=keep_set, dims=dims_222)
        slow = partial_trace_slow(rho3, keep_set, dims_222)
        _assert_close(fast, slow, f"PT fast==slow for keep={keep_set}")

    # 7f) Nieuw: niet-uniform (2,3,2), vormcontrole voor verschillende keep-sets
    dims_232 = [2,3,2]
    v232 = rng.normal(size=12) + 1j*rng.normal(size=12)
    v232 /= np.linalg.norm(v232)
    rho232 = to_density(v232)
    shapes_expected = {
        (0,): (2,2), (1,): (3,3), (2,): (2,2),
        (0,1): (6,6), (1,2): (6,6), (0,2): (4,4),
        (): (1,1)
    }
    for ks, expected in shapes_expected.items():
        out = partial_trace(rho232, keep=ks, dims=dims_232)
        ok_shape = (out.shape == expected)
        print(f"[{'PASS' if ok_shape else 'FAIL'}] shape keep={ks}: got {out.shape}, expected {expected}")

    # 8) Amplitude damping-demo op |1>
    ok, tp, cp, evals = is_cptp(amplitude_damping_kraus(0.25))
    print("Amplitude damping CPTP:", ok, " (TP:", tp, ", CP:", cp, ") eigenwaarden Choi:", evals)
    rho1 = to_density(ket1)
    for g in (0.0, 0.3, 0.7, 1.0):
        rout = amplitude_damp_rho(rho1, g)
        vec = bloch_vector_from_rho(rout)
        print(f"gamma={g}  Bloch: {vec}")
