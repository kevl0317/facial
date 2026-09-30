"""Web GUI: live camera analysis (phone or PC browser) and video-file jobs.

    python -m facial serve            # HTTPS on :8443, prints a phone URL + QR code

The browser runs MediaPipe itself (WASM/WebGL), so real-time tracking happens on
the phone; only compact per-window measurements come back here to be judged.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import mimetypes
import secrets
import shutil
import socket
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .judge import DEFAULT_MODEL
from .live import SessionStore
from .models import DEFAULT_MODEL_DIR, ensure_models, ensure_web_vendor
from .pipeline import STAGES, Options, run

WEB_DIR = Path(__file__).resolve().parent / "web"
STATE_DIR = Path(__file__).resolve().parent.parent / ".facial"

# Module scripts and WebAssembly must be served with these exact types.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")


# --------------------------------------------------------------------------- video-file jobs

@dataclass
class Job:
    id: str
    workdir: Path
    options: Options
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    error: str = ""

    def report(self, stage: str, frac: float | None, msg: str) -> None:
        self.stage = stage
        if frac is not None and stage in STAGES:
            lo, hi = STAGES[stage]
            self.progress = max(self.progress, lo + (hi - lo) * frac)
        if msg:
            self.log = (self.log + msg.splitlines())[-60:]

    def public(self) -> dict:
        out = {"id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 3),
               "log": self.log[-12:], "error": self.error}
        if self.status == "done":
            out["video"] = f"/api/jobs/{self.id}/video"
            out["analysis"] = f"/api/jobs/{self.id}/analysis"
        return out


class JobRunner:
    """Runs one video job at a time in a background thread (MediaPipe is CPU-heavy)."""

    def __init__(self, root: Path):
        self.root = root
        self.jobs: dict[str, Job] = {}
        self._gate = threading.Lock()

    def submit(self, job: Job) -> None:
        self.jobs[job.id] = job
        threading.Thread(target=self._work, args=(job,), daemon=True).start()

    def _work(self, job: Job) -> None:
        with self._gate:
            job.status = "running"
            try:
                run(job.options, job.report)
                job.progress, job.status = 1.0, "done"
            except Exception as exc:  # surfaced to the GUI
                traceback.print_exc()
                job.status, job.error = "error", f"{exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------- app

class SessionIn(BaseModel):
    lang: str = "en"
    context: str = ""


class WindowIn(BaseModel):
    start: float
    end: float
    aspect: float = 16 / 9
    samples: list[dict]
    audio: str | None = None  # base64 little-endian int16 mono PCM
    audio_sr: int = 16000
    audio_t0: float | None = None
    transcript: list[dict] = []


def create_app(token: str | None = None, judge: str = "claude", model: str = DEFAULT_MODEL,
               effort: str = "medium", live_effort: str = "low", phone_url: str = "",
               state_dir: Path = STATE_DIR) -> FastAPI:
    app = FastAPI(title="facial", version=__version__, docs_url=None, redoc_url=None)
    sessions = SessionStore()
    jobs = JobRunner(state_dir / "jobs")

    def require_key(request: Request) -> None:
        if token and token not in (request.headers.get("x-facial-key"), request.query_params.get("k")):
            raise HTTPException(401, "Missing or wrong access key: open the URL printed by the server.")

    auth = [Depends(require_key)]

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/info", dependencies=auth)
    def info():
        try:
            import faster_whisper  # noqa: F401
            whisper = True
        except ImportError:
            whisper = False
        return {"version": __version__, "judge": judge, "model": model, "phone_url": phone_url,
                "whisper": whisper}

    @app.get("/api/qr.svg", dependencies=auth)
    def qr(data: str):
        try:
            import qrcode
            import qrcode.image.svg
        except ImportError:
            raise HTTPException(404, "pip install qrcode to show a QR code") from None
        img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
        return Response(img.to_string(encoding="unicode"), media_type="image/svg+xml")

    # -- live ------------------------------------------------------------------
    @app.post("/api/live/sessions", dependencies=auth)
    def new_session(body: SessionIn):
        lang = body.lang if body.lang in ("en", "zh") else "en"
        s = sessions.create(lang=lang, context=body.context[:500], judge=judge, model=model, effort=live_effort,
                            log=lambda m: print(m, file=sys.stderr))
        return {"id": s.id, "judge": judge}

    @app.post("/api/live/sessions/{sid}/windows", dependencies=auth)
    def add_window(sid: str, body: WindowIn):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "Unknown or expired session")
        if not (0 <= body.start < body.end) or body.end - body.start > 120:
            raise HTTPException(422, "Window must be 0 <= start < end, at most 120 s long")
        return session.add_window(body.start, body.end, body.samples, body.aspect, body.audio, body.audio_sr,
                                  body.audio_t0, body.transcript)

    @app.delete("/api/live/sessions/{sid}", dependencies=auth)
    def end_session(sid: str):
        sessions.delete(sid)
        return {"ok": True}

    # -- video files -------------------------------------------------------------
    @app.post("/api/jobs", dependencies=auth)
    def new_job(video: Annotated[UploadFile, File()], srt: Annotated[UploadFile | None, File()] = None,
                lang: Annotated[str, Form()] = "en", context: Annotated[str, Form()] = "",
                window: Annotated[float, Form()] = 5.0, start: Annotated[float, Form()] = 0.0,
                end: Annotated[float | None, Form()] = None, whisper: Annotated[str, Form()] = "",
                skeleton: Annotated[bool, Form()] = False):
        job_id = uuid.uuid4().hex[:12]
        workdir = jobs.root / job_id
        workdir.mkdir(parents=True)
        suffix = Path(video.filename or "input.mp4").suffix.lower() or ".mp4"
        src = workdir / f"input{suffix}"
        with open(src, "wb") as fh:
            shutil.copyfileobj(video.file, fh)
        srt_path = None
        if srt is not None and srt.filename:
            srt_path = workdir / ("subtitles" + (Path(srt.filename).suffix.lower() or ".srt"))
            with open(srt_path, "wb") as fh:
                shutil.copyfileobj(srt.file, fh)
        opts = Options(video=src, output=workdir / "annotated.mp4", srt=srt_path, whisper=whisper or None,
                       start=max(0.0, start), end=end if end and end > start else None,
                       lang=lang if lang in ("en", "zh") else "en", judge=judge, model=model, effort=effort,
                       context=context[:500], window=min(20.0, max(2.0, window)), skeleton=skeleton,
                       workdir=workdir / "work")
        job = Job(id=job_id, workdir=workdir, options=opts)
        jobs.submit(job)
        return job.public()

    def get_job(job_id: str) -> Job:
        job = jobs.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    @app.get("/api/jobs/{job_id}", dependencies=auth)
    def job_status(job_id: str):
        return get_job(job_id).public()

    @app.get("/api/jobs/{job_id}/video", dependencies=auth)
    def job_video(job_id: str):
        job = get_job(job_id)
        if job.status != "done":
            raise HTTPException(409, "Job not finished")
        return FileResponse(job.workdir / "annotated.mp4", media_type="video/mp4",
                            filename=f"facial_{job_id}.mp4", content_disposition_type="inline")

    @app.get("/api/jobs/{job_id}/analysis", dependencies=auth)
    def job_analysis(job_id: str):
        job = get_job(job_id)
        if job.status != "done":
            raise HTTPException(409, "Job not finished")
        return FileResponse(job.workdir / "annotated_analysis.json", media_type="application/json")

    # -- static ------------------------------------------------------------------
    # /models serves the .task files and the vendored MediaPipe web runtime (models/web/).
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    app.mount("/models", StaticFiles(directory=DEFAULT_MODEL_DIR, check_dir=False), name="models")
    return app


# --------------------------------------------------------------------------- serving

def lan_ips() -> list[str]:
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))  # no packets are sent; this just picks the outbound interface
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for *_, addr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(addr[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def ensure_cert(cert_dir: Path, hosts: list[str]) -> tuple[Path, Path]:
    """Self-signed certificate for localhost + LAN IPs (phones need HTTPS for the camera)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_dir.mkdir(parents=True, exist_ok=True)
    key_path, cert_path, meta = cert_dir / "key.pem", cert_dir / "cert.pem", cert_dir / "hosts.json"
    if cert_path.exists() and key_path.exists() and meta.exists() and json.loads(meta.read_text()) == hosts:
        return cert_path, key_path

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "facial local server")])
    alt_names: list[x509.GeneralName] = []
    for host in hosts:
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            alt_names.append(x509.DNSName(host))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    meta.write_text(json.dumps(hosts))
    return cert_path, key_path


def _warm_up() -> None:
    """librosa compiles its pitch tracker on first use (seconds); do it before the first live window."""
    import numpy as np

    from .audio import analyze_samples

    t = np.arange(16000) / 16000
    analyze_samples((0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), 16000)


def _print_qr(url: str) -> None:
    try:
        import qrcode
    except ImportError:
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.print_ascii(invert=True)


def serve(host: str = "0.0.0.0", port: int | None = None, https: bool = True, token: str | None = None,
          judge: str = "claude", model: str = DEFAULT_MODEL, effort: str = "medium",
          live_effort: str = "low") -> None:
    import uvicorn

    ensure_models()
    ensure_web_vendor()
    port = port or (8443 if https else 8000)
    token = token if token is not None else secrets.token_urlsafe(8)
    scheme = "https" if https else "http"
    ips = lan_ips() if host in ("0.0.0.0", "::") else [host]
    query = f"/?k={token}" if token else "/"
    local_url = f"{scheme}://localhost:{port}{query}"
    phone_url = f"{scheme}://{ips[0]}:{port}{query}" if ips else ""

    ssl = {}
    if https:
        cert, key = ensure_cert(STATE_DIR / "certs", ["localhost", "127.0.0.1", *ips])
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}

    app = create_app(token=token, judge=judge, model=model, effort=effort, live_effort=live_effort,
                     phone_url=phone_url)
    threading.Thread(target=_warm_up, daemon=True).start()
    print(f"\n  facial {__version__}  ·  judge: {judge} ({model})\n", file=sys.stderr)
    print(f"  On this computer:  {local_url}", file=sys.stderr)
    if phone_url:
        print(f"  On your phone:     {phone_url}   (same Wi-Fi)\n", file=sys.stderr)
        _print_qr(phone_url)
    if https:
        print("  The certificate is self-signed: accept the browser warning once "
              "(Advanced -> Proceed / Show details -> visit this website).\n", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="warning", **ssl)
