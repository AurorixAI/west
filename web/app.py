"""Team website and live demo.

    uvicorn web.app:app --host 0.0.0.0 --port 7860

Serves the static site (web/static) and a small job API for the live demo:
    POST /api/jobs            multipart upload "file" (.mp4) -> {"id": ...}
    POST /api/uploads         start a chunked upload -> {"id": ...}
    PUT  /api/uploads/{id}?offset=N   raw bytes of one chunk
    POST /api/uploads/{id}/done?name=  queue the assembled file as a job -> {"id": ...}
    GET  /api/jobs/{id}       state, progress, stage, and the result when done
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
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


# Chunked upload for large videos: hosts cap one request's body and duration (Modal: 4 GiB,
# 150 s), and the organisers' 4K samples are 5-6 GB. The page sends the file in pieces.
uploads: dict[str, Path] = {}


@app.post("/api/uploads")
def new_upload() -> dict:
    uid = uuid.uuid4().hex[:12]
    uploads[uid] = Path(tempfile.mkstemp(suffix=".mp4")[1])
    return {"id": uid}


@app.put("/api/uploads/{uid}")
async def upload_chunk(uid: str, offset: int, request: Request) -> dict:
    path = uploads.get(uid)
    if path is None:
        raise HTTPException(404, "unknown upload")
    if offset > path.stat().st_size:
        raise HTTPException(409, "chunk out of order")
    with path.open("r+b") as f:                   # writing at the offset makes a retried chunk harmless
        f.seek(offset)
        async for chunk in request.stream():
            f.write(chunk)
        f.truncate()
        size = f.tell()
    if size > MAX_MB * 2**20:
        uploads.pop(uid, None)
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"file is larger than {MAX_MB} MB")
    return {"size": size}


@app.post("/api/uploads/{uid}/done")
def finish_upload(uid: str, name: str) -> dict:
    path = uploads.pop(uid, None)
    if path is None:
        raise HTTPException(404, "unknown upload")
    try:
        return {"id": jobs.submit(path, name)}
    except (RuntimeError, ValueError) as exc:
        path.unlink(missing_ok=True)
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
