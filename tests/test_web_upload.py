"""The demo's chunked upload reassembles the file exactly, even when a chunk is sent twice."""
from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import HTTPException

from web import app as web


class Body:
    """Stands in for the request: the handler only reads its byte stream."""

    def __init__(self, data: bytes):
        self.data = data

    async def stream(self):
        yield self.data


def put(uid: str, offset: int, data: bytes) -> dict:
    return asyncio.run(web.upload_chunk(uid, offset, Body(data)))


def test_chunked_upload_reassembles_the_file(monkeypatch):
    queued = {}
    monkeypatch.setattr(web.jobs, "submit", lambda path, name: queued.update(data=path.read_bytes(), name=name) or "j1")
    data = os.urandom(10_000)

    uid = web.new_upload()["id"]
    assert put(uid, 0, data[:4000]) == {"size": 4000}
    assert put(uid, 0, data[:4000]) == {"size": 4000}            # a retried chunk is harmless
    with pytest.raises(HTTPException) as gap:
        put(uid, 9000, data[9000:])
    assert gap.value.status_code == 409
    assert put(uid, 4000, data[4000:]) == {"size": 10_000}
    assert web.finish_upload(uid, "clip.mp4") == {"id": "j1"}

    assert queued == {"data": data, "name": "clip.mp4"}
    with pytest.raises(HTTPException) as gone:
        web.finish_upload(uid, "clip.mp4")
    assert gone.value.status_code == 404
