"""
Bell-state runner for Quantum Math Lab.

Flow:
1) Build a 2-qubit Bell state |Phi+> = (|00> + |11>) / sqrt(2)
2) Simulate it locally
3) Run it on real IBM Quantum hardware via runtime API ("qcapi" path)

Typical usage in Ubuntu/WSL with your requested venv:
    source ~/.venvs/qiskit/bin/activate
    python bell_qcapi_runner.py --mode both --shots 1024

Optional:
    python bell_qcapi_runner.py --mode hardware --backend ibm_torino --shots 2048
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class HardwareRun:
    backend_name: str
    job_id: str
    counts: Dict[str, int]


def make_bell_circuit() -> Any:
    """Create a Bell circuit that measures both qubits in Z basis."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit is niet beschikbaar. Activeer je qiskit-venv of installeer qiskit."
        ) from exc

    qc = QuantumCircuit(2, 2, name="bell_phi_plus")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _backend_name(backend: Any) -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr())
        except TypeError:
            pass
    if name_attr is not None:
        return str(name_attr)
    return str(backend)


def _normalize_counts(counts: Mapping[Any, Any], num_bits: int = 2) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{num_bits}b")
        else:
            bitstring = str(key).replace(" ", "")
        out[bitstring] = int(round(float(value)))
    return out


def _quasi_to_counts(quasi: Mapping[Any, Any], shots: int, num_bits: int = 2) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for key, prob in quasi.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{num_bits}b")
        else:
            bitstring = str(key).replace(" ", "")
        counts[bitstring] = int(round(float(prob) * shots))
    return counts


def _extract_counts_from_sampler_result(result: Any, shots: int) -> Dict[str, int]:
    """Handle multiple qiskit-runtime result shapes (Sampler v1/v2)."""
    if hasattr(result, "quasi_dists"):
        quasi_dists = getattr(result, "quasi_dists")
        if quasi_dists:
            return _quasi_to_counts(quasi_dists[0], shots=shots)

    first = None
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
    elif hasattr(result, "__getitem__"):
        try:
            first = result[0]
        except Exception:
            first = None

    if first is not None:
        data = getattr(first, "data", None)
        if data is not None:
            for register_name in ("c", "meas"):
                register = getattr(data, register_name, None)
                if register is not None and hasattr(register, "get_counts"):
                    return _normalize_counts(register.get_counts())
            if hasattr(data, "get_counts"):
                return _normalize_counts(data.get_counts())

    if hasattr(result, "get_counts"):
        return _normalize_counts(result.get_counts())

    raise RuntimeError(
        "Kon counts niet uit runtime-resultaat halen. "
        "Controleer qiskit/qiskit-ibm-runtime versie."
    )


def _bell_parity(counts: Mapping[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    return (counts.get("00", 0) + counts.get("11", 0)) / total


def run_simulation(shots: int) -> tuple[str, Dict[str, int]]:
    """Run Bell circuit on local simulator."""
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise RuntimeError("Qiskit import faalde tijdens simulatie.") from exc

    backend = None
    backend_label = ""

    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        backend_label = "AerSimulator"
    except Exception:
        try:
            from qiskit.providers.basic_provider import BasicSimulator

            backend = BasicSimulator()
            backend_label = "BasicSimulator"
        except Exception:
            try:
                from qiskit import BasicAer

                backend = BasicAer.get_backend("qasm_simulator")
                backend_label = "BasicAer/qasm_simulator"
            except Exception as exc:
                raise RuntimeError(
                    "Geen lokale simulator gevonden. Installeer qiskit-aer of update qiskit."
                ) from exc

    circuit = make_bell_circuit()
    transpiled = transpile(circuit, backend=backend, optimization_level=1)
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = _normalize_counts(result.get_counts())
    return backend_label, counts


def _build_runtime_service() -> Any:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError(
            "qiskit-ibm-runtime ontbreekt. Installeer het in je qiskit-venv."
        ) from exc

    token = (
        os.getenv("QCAPI_TOKEN")
        or os.getenv("QISKIT_IBM_TOKEN")
        or os.getenv("IBM_QUANTUM_TOKEN")
    )
    instance = os.getenv("QISKIT_IBM_INSTANCE")

    if token:
        kwargs: Dict[str, str] = {"token": token}
        if instance:
            kwargs["instance"] = instance
        try:
            return QiskitRuntimeService(channel="ibm_quantum", **kwargs)
        except TypeError:
            return QiskitRuntimeService(**kwargs)

    return QiskitRuntimeService()


def _select_backend(service: Any, requested_backend: str | None) -> Any:
    if requested_backend:
        return service.backend(requested_backend)

    candidates = service.backends(simulator=False, operational=True, min_num_qubits=2)
    if not candidates:
        candidates = service.backends(simulator=False, min_num_qubits=2)
    if not candidates:
        raise RuntimeError("Geen geschikte hardware-backends gevonden.")

    try:
        from qiskit_ibm_runtime import least_busy

        return least_busy(candidates)
    except Exception:
        scored = []
        for backend in candidates:
            pending = 10**9
            try:
                pending = int(backend.status().pending_jobs)
            except Exception:
                pass
            scored.append((pending, _backend_name(backend), backend))
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2]


def list_backends() -> None:
    service = _build_runtime_service()
    backends = service.backends(simulator=False, operational=True, min_num_qubits=2)
    print("Beschikbare hardware-backends (>=2 qubits):")
    for backend in sorted(backends, key=_backend_name):
        print(f"- {_backend_name(backend)}")


def run_hardware(shots: int, backend_name: str | None) -> HardwareRun:
    try:
        from qiskit import transpile
    except ImportError as exc:
        raise RuntimeError("Qiskit import faalde tijdens hardware-run.") from exc

    service = _build_runtime_service()
    backend = _select_backend(service, backend_name)

    circuit = make_bell_circuit()
    transpiled = transpile(circuit, backend=backend, optimization_level=1)

    job = None
    result = None
    try:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=backend)
        job = sampler.run([transpiled], shots=shots)
        result = job.result()
    except ImportError:
        from qiskit_ibm_runtime import Sampler, Session

        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()

    if job is None or result is None:
        raise RuntimeError("Hardware job kon niet worden gestart.")

    counts = _extract_counts_from_sampler_result(result, shots=shots)
    return HardwareRun(
        backend_name=_backend_name(backend),
        job_id=str(job.job_id()),
        counts=counts,
    )


def _print_counts(title: str, counts: Mapping[str, int]) -> None:
    print(f"{title}: {json.dumps(dict(sorted(counts.items())), ensure_ascii=True)}")
    print(f"{title} parity P(00)+P(11): {_bell_parity(counts):.4f}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maak Bell-state, simuleer en run op quantum hardware via qcapi/runtime."
    )
    parser.add_argument(
        "--mode",
        choices=("simulate", "hardware", "both"),
        default="both",
        help="Kies alleen simulatie, alleen hardware, of beide (default: both).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Aantal shots voor simulatie/hardware run.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optionele backendnaam, bv. ibm_torino. Leeg = automatisch least-busy.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Toon beschikbare hardware-backends en stop.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.shots <= 0:
        print("Shots moet > 0 zijn.", file=sys.stderr)
        return 2

    if args.list_backends:
        try:
            list_backends()
            return 0
        except Exception as exc:
            print(f"Backend-lijst ophalen faalde: {exc}", file=sys.stderr)
            return 1

    print("Bell circuit:")
    print(make_bell_circuit().draw(output="text"))

    success = True

    if args.mode in ("simulate", "both"):
        try:
            backend_label, sim_counts = run_simulation(shots=args.shots)
            _print_counts(f"Simulatie ({backend_label})", sim_counts)
        except Exception as exc:
            success = False
            print(f"Simulatie faalde: {exc}", file=sys.stderr)

    if args.mode in ("hardware", "both"):
        try:
            hw = run_hardware(shots=args.shots, backend_name=args.backend)
            print(f"Hardware backend: {hw.backend_name}")
            print(f"Hardware job id: {hw.job_id}")
            _print_counts("Hardware counts", hw.counts)
        except Exception as exc:
            success = False
            print(f"Hardware run faalde: {exc}", file=sys.stderr)
            print(
                "Tip: zet een IBM token in QCAPI_TOKEN of QISKIT_IBM_TOKEN, "
                "of save account via QiskitRuntimeService.save_account(...).",
                file=sys.stderr,
            )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
