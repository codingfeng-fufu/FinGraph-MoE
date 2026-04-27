from __future__ import annotations

from pathlib import Path
import sys
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_router.realdata_demo import RealdataDemoService


APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"

app = FastAPI(title="MoE Real-Data Demo")
_service: RealdataDemoService | None = None
_service_error: str | None = None

DEFAULT_EXAMPLES = [
    "计算 LIFE INSURANCE 行业公司的 revenue 总和",
    "请统计 2024 年 revenue >= 1400 的公司数量",
    "请预测 Company_3 在 2024 年的收入",
]


def get_service() -> RealdataDemoService:
    global _service, _service_error
    if _service is not None:
        return _service
    try:
        _service = RealdataDemoService("data/moe_router_realdata_v1")
        _service_error = None
        return _service
    except Exception as exc:
        _service_error = str(exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Demo artifacts are not configured. Build or provide router/model artifacts "
                f"under data/ and runs/ before using expert inference. Details: {_service_error}"
            ),
        ) from exc


class DemoRequest(BaseModel):
    query: str
    top_k: int = 7
    session_id: str | None = None


class DocumentRequest(BaseModel):
    file_name: str
    media_type: str = "application/octet-stream"
    content_base64: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/workbench")
def workbench() -> FileResponse:
    return FileResponse(STATIC_DIR / "workbench.html")


@app.get("/api/examples")
def examples() -> dict[str, list[str]]:
    try:
        return {"examples": get_service().examples()}
    except HTTPException:
        return {"examples": DEFAULT_EXAMPLES}


@app.get("/api/health")
def health() -> dict[str, object]:
    try:
        get_service()
        return {"ready": True, "service_error": None}
    except HTTPException:
        return {"ready": False, "service_error": _service_error}


@app.post("/api/document/process")
def process_document(request: DocumentRequest) -> dict:
    content = base64.b64decode(request.content_base64.encode("utf-8"))
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return get_service().create_document_session(
            file_name=request.file_name or "document",
            media_type=request.media_type or "application/octet-stream",
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/document/{session_id}")
def get_document(session_id: str) -> dict:
    try:
        return get_service().get_document_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


@app.post("/api/run")
def run_demo(request: DemoRequest) -> dict:
    try:
        return get_service().run(request.query, top_k=request.top_k, session_id=request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
