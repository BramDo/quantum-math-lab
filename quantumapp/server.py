"""
Quantum Web App (NumPy-only, stdlib WSGI)
----------------------------------------

Kleine webapp met vergelijkbare mogelijkheden als de CLI:
- Zelftests uit `quant_math_lab`
- Playground-demos uit `qcqi_pure_math_playground.py`
- CHSH-waarde (Bell-staat, optioneel custom hoeken)
- QPE-simulatie (theta, m)
- Notebook-uitvoering (.ipynb) via bestaande runner

Afhankelijkheden: alleen standaardbibliotheek + NumPy (zoals repo-richtlijnen).

Start:
  python -m quantumapp.server --host 127.0.0.1 --port 8000
  (of) python quantumapp/server.py

Open dan http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple

import numpy as np
from wsgiref.simple_server import make_server
from wsgiref.util import request_uri


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ROOT_DIR = os.path.dirname(BASE_DIR)


def _json_response(start_response, obj: Dict[str, Any], status: str = "200 OK"):
    data = json.dumps(obj).encode("utf-8")
    start_response(status, [("Content-Type", "application/json; charset=utf-8"),
                            ("Content-Length", str(len(data))),
                            ("Cache-Control", "no-store")])
    return [data]


def _read_json(environ) -> Dict[str, Any]:
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        length = 0
    body = environ["wsgi.input"].read(length) if length > 0 else b""
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return {}


def _capture(fn, *args, **kwargs) -> Tuple[bool, str]:
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            fn(*args, **kwargs)
        out = buf_out.getvalue()
        err = buf_err.getvalue()
        msg = out + ("\n[stderr]\n" + err if err else "")
        return True, msg.strip()
    except SystemExit:
        out = buf_out.getvalue()
        err = buf_err.getvalue()
        msg = out + ("\n[stderr]\n" + err if err else "")
        return True, msg.strip()
    except Exception:
        tb = traceback.format_exc()
        out = buf_out.getvalue()
        err = buf_err.getvalue() + ("\n" + tb)
        msg = out + ("\n[stderr]\n" + err if err else "")
        return False, msg.strip()


def handle_self_tests(start_response):
    import quant_math_lab as qml

    def run():
        qml.run_self_tests(verbose=True)

    ok, msg = _capture(run)
    return _json_response(start_response, {"ok": ok, "output": msg})


def handle_playground(start_response):
    import runpy

    def run():
        runpy.run_path(os.path.join(ROOT_DIR, "qcqi_pure_math_playground.py"), run_name="__main__")

    ok, msg = _capture(run)
    return _json_response(start_response, {"ok": ok, "output": msg})


def _bell_state(state: str, qml) -> np.ndarray:
    k0 = qml.tensor(qml.basis_ket(2, 0), qml.basis_ket(2, 0))
    k1 = qml.tensor(qml.basis_ket(2, 1), qml.basis_ket(2, 1))
    k01 = qml.tensor(qml.basis_ket(2, 0), qml.basis_ket(2, 1))
    k10 = qml.tensor(qml.basis_ket(2, 1), qml.basis_ket(2, 0))
    if state == "phi+":
        return (k0 + k1) / np.sqrt(2)
    if state == "phi-":
        return (k0 - k1) / np.sqrt(2)
    if state == "psi+":
        return (k01 + k10) / np.sqrt(2)
    if state == "psi-":
        return (k01 - k10) / np.sqrt(2)
    raise ValueError("state ∈ {phi+,phi-,psi+,psi-}")


def handle_chsh(environ, start_response):
    data = _read_json(environ)
    state = data.get("state", "phi+")
    a0 = data.get("a0")
    a1 = data.get("a1")
    b0 = data.get("b0")
    b1 = data.get("b1")
    try:
        import quant_math_lab as qml
        psi = _bell_state(state, qml)
        if any(v is not None for v in (a0, a1, b0, b1)):
            # Custom hoeken via qcqi helper
            a0 = float(a0 if a0 is not None else 0.0)
            a1 = float(a1 if a1 is not None else (np.pi/2))
            b0 = float(b0 if b0 is not None else (np.pi/4))
            b1 = float(b1 if b1 is not None else (-np.pi/4))
            try:
                import qcqi_pure_math_playground as qcqi
                val = float(qcqi.chsh_value(psi.reshape(-1), a0, a1, b0, b1))
                return _json_response(start_response, {"ok": True, "value": val,
                                                      "angles": {"a0": a0, "a1": a1, "b0": b0, "b1": b1}})
            except Exception:
                pass
        # Fallback: standaardbasis-meting in qml
        val = float(np.real(qml.chsh_value(qml.dm(psi))))
        return _json_response(start_response, {"ok": True, "value": val})
    except Exception as e:
        return _json_response(start_response, {"ok": False, "error": str(e)}, status="400 Bad Request")


def handle_qpe(environ, start_response):
    data = _read_json(environ)
    theta = float(data.get("theta", 0.34375))
    m = int(data.get("m", 6))
    try:
        import quant_math_lab as qml
        U = np.diag([1.0 + 0.0j, np.exp(2j * np.pi * theta)])
        psi = qml.basis_ket(2, 1)
        y, approx = qml.qpe_simulate(U, psi, m=m)
        return _json_response(start_response, {"ok": True, "y": int(y), "approx": float(approx), "theta": theta, "m": m})
    except Exception as e:
        return _json_response(start_response, {"ok": False, "error": str(e)}, status="400 Bad Request")


def handle_notebook(environ, start_response):
    data = _read_json(environ)
    path = data.get("path")
    if not path:
        return _json_response(start_response, {"ok": False, "error": "path vereist"}, status="400 Bad Request")
    abspath = os.path.abspath(path if os.path.isabs(path) else os.path.join(ROOT_DIR, path))
    # Beperk tot repo-root
    if not abspath.startswith(ROOT_DIR) or not abspath.endswith('.ipynb'):
        return _json_response(start_response, {"ok": False, "error": "ongeldig pad"}, status="400 Bad Request")
    try:
        import quantum_app as qa

        def run():
            qa.run_notebook(abspath)

        ok, msg = _capture(run)
        return _json_response(start_response, {"ok": ok, "output": msg, "path": abspath})
    except Exception as e:
        return _json_response(start_response, {"ok": False, "error": str(e)}, status="400 Bad Request")


def serve_static(path: str, start_response):
    # Alleen index.html en eenvoudige assets vanuit STATIC_DIR
    full = os.path.join(STATIC_DIR, path)
    if not os.path.abspath(full).startswith(STATIC_DIR) or not os.path.isfile(full):
        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"Not found"]
    ctype = "text/plain; charset=utf-8"
    if full.endswith(".html"):
        ctype = "text/html; charset=utf-8"
    elif full.endswith(".css"):
        ctype = "text/css; charset=utf-8"
    elif full.endswith(".js"):
        ctype = "application/javascript; charset=utf-8"
    with open(full, "rb") as f:
        data = f.read()
    start_response("200 OK", [("Content-Type", ctype), ("Content-Length", str(len(data)))])
    return [data]


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    # Routing
    if method == "GET" and (path == "/" or path == "/index.html"):
        return serve_static("index.html", start_response)
    if method == "GET" and path.startswith("/static/"):
        return serve_static(path[len("/static/"):], start_response)

    if method == "POST" and path == "/api/self-tests":
        return handle_self_tests(start_response)
    if method == "POST" and path == "/api/playground":
        return handle_playground(start_response)
    if method == "POST" and path == "/api/chsh":
        return handle_chsh(environ, start_response)
    if method == "POST" and path == "/api/qpe":
        return handle_qpe(environ, start_response)
    if method == "POST" and path == "/api/notebook":
        return handle_notebook(environ, start_response)

    start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
    return [f"Not found: {path}".encode("utf-8")]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quantum Web App (stdlib WSGI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    with make_server(args.host, args.port, app) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving on {url}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()

