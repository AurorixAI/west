"""Team website and live demo.

    uvicorn web.app:app --host 0.0.0.0 --port 7860

Serves the static site (web/static) and a small job API for the live demo:
    POST /api/jobs            multipart upload "file" (.mp4) -> {"id": ...}
    GET  /api/jobs/{id}       state, progress, stage, and the result when done
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from web.demo import MAX_MB, MAX_SEC, Jobs  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Team WEST: traffic event detection")
jobs = Jobs()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "max_mb": MAX_MB, "max_sec": MAX_SEC}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(400, "please upload an .mp4 file")
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    size = 0
    with tmp.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB * 2**20:
                f.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"file is larger than {MAX_MB} MB")
            f.write(chunk)
    try:
        return {"id": jobs.submit(tmp, file.filename)}
    except (RuntimeError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))


@app.get("/api/jobs/{jid}")
def job(jid: str) -> JSONResponse:
    j = jobs.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job (results are kept for two hours)")
    return JSONResponse(j)


@app.get("/predictions_samples.json")
def predictions() -> FileResponse:
    path = ROOT / "predictions_samples.json"
    if not path.exists():
        raise HTTPException(404, "predictions_samples.json has not been generated in this deployment")
    return FileResponse(path, media_type="application/json")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
