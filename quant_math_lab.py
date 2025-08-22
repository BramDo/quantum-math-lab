
"""
quant_math_lab.py
-----------------

Een compacte "werkplaats" voor de wiskunde achter kwantummechanica en
kwantuminformatie: Dirac-notatie ↔ matrices, unitariteit, projectoren,
dichtheidsmatrices, partiële spoor (partial trace), commutatoren,
onzekerheidsrelaties, CHSH-waarde, eenvoudige orakels (Deutsch/Deutsch–Jozsa)
en een pure NumPy-simulatie van Quantum Phase Estimation (QPE).

Alles is bewust dependency‑light (NumPy only). Gebruik dit bestand om te leren
én te testen: elk blok heeft korte docstrings en er staat onderaan een
`__main__` met assert‑tests die je kunt draaien.

Usage
-----
>>> import numpy as np
>>> from quant_math_lab import H, X, Z, ket, dm, is_unitary
>>> is_unitary(H)
True
>>> psi = ket([1, 0])  # |0>
>>> rho = dm(psi)
>>> float(rho.trace().real)
1.0

Tip: bekijk ook de `run_self_tests()` functie onderaan.

(c) 2025 — MIT License
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, List
import math
import cmath
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


# -----------------------------
# Basis lineaire-algebra helpers
# -----------------------------

def ket(x: Sequence[complex]) -> Array:
    """Maak een gekwadrateerde-kolomvector (ket) en normaliseer.

    >>> v = ket([1, 1])
    >>> np.allclose(v.conj().T @ v, 1.0)
    True
    """
    v = np.asarray(x, dtype=np.complex128).reshape(-1, 1)
    nrm = np.linalg.norm(v)
    return v if nrm == 0 else v / nrm


def bra(v: Array) -> Array:
    """Bra van een ket of (mogelijk) 1D-array.

    >>> v = ket([1, 0j])
    >>> b = bra(v)
    >>> b.shape
    (1, 2)
    """
    v = np.asarray(v, dtype=np.complex128).reshape(-1, 1)
    return v.conj().T


def dagger(M: Array) -> Array:
    """Conjugaat-getransponeerd (†)."""
    return np.asarray(M, dtype=np.complex128).conj().T


def outer(u: Array, v: Array) -> Array:
    """|u><v| (met u en v als kolomvectoren).

    >>> u, v = ket([1,0]), ket([0,1])
    >>> M = outer(u, v)
    >>> np.allclose(M, np.array([[0,1],[0,0]],dtype=complex))
    True
    """
    u = np.asarray(u, dtype=np.complex128).reshape(-1, 1)
    v = np.asarray(v, dtype=np.complex128).reshape(-1, 1)
    return u @ v.conj().T


def tensor(*ops: Array) -> Array:
    """Kronecker-product van links naar rechts.

    >>> from quant_math_lab import X, Z
    >>> np.allclose(tensor(X, Z), np.kron(X, Z))
    True
    """
    out = np.array([[1.0+0.0j]])
    for op in ops:
        out = np.kron(out, np.asarray(op, dtype=np.complex128))
    return out


def is_unitary(U: Array, atol: float = 1e-10) -> bool:
    """Check U†U = I."""
    U = np.asarray(U, dtype=np.complex128)
    I = np.eye(U.shape[0], dtype=np.complex128)
    return np.allclose(dagger(U) @ U, I, atol=atol)


def projector(psi: Array) -> Array:
    """Projector |psi><psi| (psi moet genormaliseerd zijn).

    >>> p = projector(ket([1,0]))
    >>> np.allclose(p, np.array([[1,0],[0,0]],complex))
    True
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1, 1)
    return outer(psi, psi)


def basis_ket(dim: int, idx: int) -> Array:
    """|idx> in gegeven dimensie.

    >>> basis_ket(3,2).shape
    (3, 1)
    """
    e = np.zeros((dim, 1), dtype=np.complex128)
    e[idx, 0] = 1.0
    return e


# -----------------
# Elementaire poorten
# -----------------

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = (1/np.sqrt(2))*np.array([[1, 1], [1, -1]], dtype=np.complex128)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128)


def Rx(theta: float) -> Array:
    c, s = np.cos(theta/2), -1j*np.sin(theta/2)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def Ry(theta: float) -> Array:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def Rz(theta: float) -> Array:
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex128)


def CNOT() -> Array:
    """CNOT met control qubit 0 (MSB) en target qubit 1 (LSB)."""
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=np.complex128)


def SWAP() -> Array:
    return np.array([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]], dtype=np.complex128)


def controlled(U: Array) -> Array:
    """Bouw een 2‑qubit controlled‑U met control op qubit 0.

    >>> CUx = controlled(X)
    >>> is_unitary(CUx)
    True
    """
    U = np.asarray(U, dtype=np.complex128)
    d = U.shape[0]
    I = np.eye(d, dtype=np.complex128)
    P0 = np.array([[1,0],[0,0]], dtype=np.complex128)
    P1 = np.array([[0,0],[0,1]], dtype=np.complex128)
    return tensor(P0, I) + tensor(P1, U)


# ----------------------
# Staten & dichtheidsmatrices
# ----------------------

def dm(psi: Array) -> Array:
    """Dichtheidsmatrix van pure staat |psi>.

    >>> rho = dm(ket([1,0]))
    >>> float(rho.trace().real) == 1.0
    True
    """
    return projector(psi)


def born_probs(rho_or_psi: Array) -> np.ndarray:
    """Born‑kansen bij standaardbasis-meting van rho of psi.

    >>> p = born_probs(ket([1,0]))
    >>> np.allclose(p, [1,0])
    True
    """
    A = np.asarray(rho_or_psi, dtype=np.complex128)
    if A.ndim == 2 and A.shape[1] == 1:  # ket
        A = dm(A)
    return np.real(np.diag(A))


def partial_trace(rho: Array, dims: Sequence[int], traced_out: Sequence[int]) -> Array:
    """Partiële spoor over subsystemen in `traced_out`.

    Args:
        rho: dichtheidsmatrix op (∏ dims) x (∏ dims)
        dims: lokale dimensies, bv. [2,2,3]
        traced_out: indices van subsystemen die je 'weggooit'

    Returns:
        Reduced density matrix over de resterende subsystemen.

    >>> # Bell‑staat |phi+> gereduceerd is I/2
    >>> phi_plus = (tensor(basis_ket(2,0), basis_ket(2,0)) + tensor(basis_ket(2,1), basis_ket(2,1)))/np.sqrt(2)
    >>> rho = dm(phi_plus)
    >>> red = partial_trace(rho, dims=[2,2], traced_out=[1])
    >>> np.allclose(red, 0.5*I2)
    True
    """
    dims = list(dims)
    keep = [i for i in range(len(dims)) if i not in traced_out]
    # herschik indices naar |keep, traced_out> basis
    dim_total = np.prod(dims)
    rho = rho.reshape([*dims, *dims])
    axes = list(range(len(dims)))
    perm = keep + list(traced_out) + [len(dims)+i for i in keep] + [len(dims)+i for i in traced_out]
    rho_perm = np.transpose(rho, perm)
    dk = int(np.prod([dims[i] for i in keep])) or 1
    dt = int(np.prod([dims[i] for i in traced_out])) or 1
    rho_perm = rho_perm.reshape(dk, dt, dk, dt)
    red = np.einsum('ijik->jk', rho_perm)  # trace over dt-as
    return red


# ----------------------
# Operator algebra & onzekerheid
# ----------------------

def commutator(A: Array, B: Array) -> Array:
    """[A,B] = AB − BA"""
    return A @ B - B @ A


def expectation(op: Array, state: Array) -> complex:
    """<op> voor pure staat of dichtheidsmatrix."""
    if state.ndim == 2 and state.shape[1] == 1:
        return (bra(state) @ op @ state)[0,0]
    return np.trace(state @ op)


def variance(op: Array, state: Array) -> float:
    mu = expectation(op, state)
    return float(np.real(expectation(op @ op, state) - mu*mu))


def uncertainty_bound(A: Array, B: Array, state: Array) -> Tuple[float, float, float]:
    r"""Geef (ΔA, ΔB, bound) met
    ΔA ΔB ≥ | <[A,B]> | / 2

    >>> # Voor X en Z in |+>:
    >>> plus = ket([1,1])
    >>> dX, dZ, b = uncertainty_bound(X, Z, plus)
    >>> dX*dZ + 1e-8 >= b
    True
    """
    dA = math.sqrt(max(0.0, variance(A, state)))
    dB = math.sqrt(max(0.0, variance(B, state)))
    comm = expectation(commutator(A, B), state)
    bound = abs(comm) / 2.0
    return dA, dB, float(np.real(bound))


# ----------------------
# CHSH-waarde
# ----------------------

def chsh_value(state: Array) -> float:
    """Bereken |S| voor standaard CHSH-keuze.
    A0 = Z, A1 = X op qubit A
    B0 = (Z + X)/√2, B1 = (Z − X)/√2 op qubit B.

    Verwachting voor |phi+>: 2√2 ≈ 2.828

    >>> phi_plus = (tensor(basis_ket(2,0), basis_ket(2,0)) + tensor(basis_ket(2,1), basis_ket(2,1)))/np.sqrt(2)
    >>> abs(chsh_value(dm(phi_plus)) - 2*np.sqrt(2)) < 1e-6
    True
    """
    A0 = tensor(Z, I2)
    A1 = tensor(X, I2)
    B0 = tensor(I2, (Z + X)/np.sqrt(2))
    B1 = tensor(I2, (Z - X)/np.sqrt(2))
    S = A0 @ (B0 + B1) + A1 @ (B0 - B1)
    return float(np.real(expectation(S, state)))


# ----------------------
# Eenvoudige orakels & algoritmes
# ----------------------

def deutsch_oracle(func_type: str) -> Array:
    """Maak U_f voor Deutsch' probleem.
    func_type in {"constant0","constant1","identity","not"}.
    U_f |x>|y> = |x>|y ⊕ f(x)>

    Returns een 4x4 unitary.
    """
    f = {
        "constant0": lambda x: 0,
        "constant1": lambda x: 1,
        "identity":  lambda x: x,
        "not":       lambda x: 1-x,
    }[func_type]
    Uf = np.zeros((4,4), dtype=np.complex128)
    for x in [0,1]:
        for y in [0,1]:
            x2 = x
            y2 = y ^ f(x)
            i = 2*x + y
            j = 2*x2 + y2
            Uf[j,i] = 1.0
    return Uf


def deutsch_algorithm(Uf: Array) -> str:
    """Implementeer Deutsch: 1 query ≈ onderscheid constant vs gebalanceerd.

    Output: "constant" of "balanced".
    """
    # |0>|1> -> H⊗H -> Uf -> H⊗I -> meet eerste qubit
    psi = tensor(ket([1,0]), ket([0,1]))  # |0>|1>
    psi = tensor(H, H) @ psi
    psi = Uf @ psi
    psi = tensor(H, I2) @ psi
    # kans op |0>~constant, |1>~balanced
    probs = np.abs(psi.reshape(4))**2
    p0 = probs[0] + probs[1]
    return "constant" if p0 > 0.5 else "balanced"


def dj_oracle(n: int, kind: str = "constant") -> Array:
    """Deutsch–Jozsa-orakel op n bits als fase-orakel Z_f (2^n x 2^n)."""
    N = 2**n
    diag = np.ones(N, dtype=np.complex128)
    if kind == "constant0":
        pass
    elif kind == "constant1":
        diag *= -1
    elif kind == "balanced":
        # neem helft +1, helft -1
        diag[:N//2] = 1.0
        diag[N//2:] = -1.0
    else:
        raise ValueError("kind ∈ {constant0, constant1, balanced}")
    return np.diag(diag)


def deutsch_jozsa(Zf: Array) -> str:
    """Voer DJ uit met fase-orakel Z_f (dim 2^n).

    Return "constant" of "balanced".
    """
    n = int(round(np.log2(Zf.shape[0])))
    psi = ket([1] + [0]*(2**n - 1))              # |0...0>
    psi = tensor(*([H]*n)) @ psi                 # Hadamards
    psi = Zf @ psi                               
    psi = tensor(*([H]*n)) @ psi
    # Als f constant: alle amplitudes nul behalve |0...0>
    amps = psi.reshape(-1)
    is_constant = np.allclose(amps[1:], 0, atol=1e-9)
    return "constant" if is_constant else "balanced"


# ----------------------
# QFT en Quantum Phase Estimation (simulatie)
# ----------------------

def qft(n: int) -> Array:
    """QFT op n qubits (dim = 2^n)."""
    N = 2**n
    omega = np.exp(2j*np.pi/N)
    M = np.fromfunction(lambda r,c: omega**(r*c), (N, N), dtype=int)
    return (1/np.sqrt(N))*M


def iqft(n: int) -> Array:
    """Inverse QFT op n qubits."""
    return dagger(qft(n))


def controlled_power(U: Array, power: int) -> Array:
    """Controlled-U^power op 1 control + d-dim target (als matrix)."""
    Up = np.linalg.matrix_power(U, power)
    d = U.shape[0]
    P0 = np.array([[1,0],[0,0]], dtype=np.complex128)
    P1 = np.array([[0,0],[0,1]], dtype=np.complex128)
    return tensor(P0, np.eye(d, dtype=np.complex128)) + tensor(P1, Up)


def qpe_simulate(U: Array, psi: Array, m: int) -> Tuple[int, float]:
    """Klassieke simulatie van QPE met m 'teller'-qubits.

    Veronderstelt dat |psi> een eigenvector is: U|psi> = e^{2π i θ} |psi>.
    Return: (y, y/2^m) als benadering van θ.
    """
    d = U.shape[0]
    # |0>^m ⊗ |psi>
    control = ket([1] + [0]*(2**m - 1))  # we houden de control expliciet als assen
    state = tensor(control, psi)          # (2^m * d, 1)
    # Hadamards op control-register (alleen op control-as toepassen)
    Hm = tensor(*([H]*m))
    state = tensor(Hm, np.eye(d, dtype=np.complex128)) @ state

    # Reshape naar (2^m, d) zodat we conditioneel U^{2^k} op rijen kunnen toepassen
    amps = state.reshape(2**m, d)

    for k in range(m):
        Up = np.linalg.matrix_power(U, 2**k)  # d x d
        # indices waarvoor de k-de bit 1 is
        for c in range(2**m):
            if (c >> k) & 1:
                amps[c, :] = (Up @ amps[c, :].reshape(d,1)).reshape(d)

    # inverse QFT op control-as
    IQ = iqft(m)  # 2^m x 2^m
    amps = (IQ @ amps)  # (2^m x d)

    # meet control
    probs = np.sum(np.abs(amps)**2, axis=1)  # som over target
    y = int(np.argmax(probs))
    return y, y/(2**m)

# ----------------------
# Zelftests
# ----------------------

def run_self_tests(verbose: bool = True) -> None:
    """Draai een serie assert‑tests. Geeft raise bij falen."""
    # Unitariteit
    assert is_unitary(H) and is_unitary(X) and is_unitary(Rz(0.123))
    # Bell‑reductie
    phi_plus = (tensor(basis_ket(2,0), basis_ket(2,0)) + tensor(basis_ket(2,1), basis_ket(2,1)))/np.sqrt(2)
    red = partial_trace(dm(phi_plus), dims=[2,2], traced_out=[1])
    assert np.allclose(red, 0.5*I2, atol=1e-10)
    # Onzekerheid
    plus = ket([1,1])
    dX, dZ, b = uncertainty_bound(X, Z, plus)
    assert dX*dZ + 1e-9 >= b
    # CHSH
    Sval = chsh_value(dm(phi_plus))
    assert abs(Sval - 2*np.sqrt(2)) < 1e-6
    # Deutsch
    for t in ["constant0","constant1","identity","not"]:
        Uf = deutsch_oracle(t)
        label = deutsch_algorithm(Uf)
        if t in ("constant0","constant1"):
            assert label == "constant"
        else:
            assert label == "balanced"
    # Deutsch–Jozsa
    Zfc = dj_oracle(4, "constant1")
    Zfb = dj_oracle(4, "balanced")
    assert deutsch_jozsa(Zfc) == "constant"
    assert deutsch_jozsa(Zfb) == "balanced"
    # QPE: neem U = diag(1, e^{2π i θ}) en eigenvector |1>
    theta = 0.34375  # 11/32 exact
    U = np.diag([1.0+0.0j, np.exp(2j*np.pi*theta)])
    y, approx = qpe_simulate(U, basis_ket(2,1), m=6)
    assert y == int(round(theta*(2**6))) or abs(approx - theta) < 1/(2**6)
    if verbose:
        print("✅ Alle zelftests geslaagd.")
        print(f"   CHSH(|phi+>) = {Sval:.6f}")
        print(f"   QPE θ ≈ {approx:.6f} (true {theta:.6f})")


if __name__ == "__main__":
    run_self_tests()
