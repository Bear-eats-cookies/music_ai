"""
FastAPI route definitions.
"""
from __future__ import annotations

import shutil
import traceback
from functools import partial
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline import MusicAIPipeline
from src.voice_cloning.rvc_runtime import discover_runtime_models


project_root = Path(__file__).resolve().parents[2]
data_root = project_root / "data"
data_root.mkdir(parents=True, exist_ok=True)

_pipeline: Optional[MusicAIPipeline] = None
_pipeline_error: Optional[str] = None

app = FastAPI(title="AI Music Generation API")

# Frontend and generated media need browser-accessible URLs.
app.mount("/static", StaticFiles(directory=str(project_root / "frontend" / "static")), name="static")
app.mount("/media", StaticFiles(directory=str(data_root)), name="media")


class GenerateRequest(BaseModel):
    audio_id: str
    style: Optional[str] = None
    lyrics: Optional[str] = None
    rvc_model_name: Optional[str] = None
    rvc_model_path: Optional[str] = None


def get_pipeline() -> MusicAIPipeline:
    """Initialize the heavy pipeline lazily so the API can start faster."""
    global _pipeline, _pipeline_error

    if _pipeline is not None:
        return _pipeline

    try:
        _pipeline = MusicAIPipeline()
        _pipeline_error = None
        return _pipeline
    except Exception as exc:
        _pipeline = None
        _pipeline_error = f"Pipeline初始化失败: {exc}"
        print(f"警告: {_pipeline_error}")
        traceback.print_exc()
        raise RuntimeError(_pipeline_error) from exc


def require_pipeline() -> MusicAIPipeline:
    """Return an initialized pipeline or surface a clear API error."""
    try:
        return get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_upload_path(audio_id: str) -> Path:
    """Resolve the uploaded audio path for a request."""
    return data_root / "uploads" / f"{audio_id}.wav"


def build_media_url(path_value: Optional[str]) -> Optional[str]:
    """Convert a local file path under `data/` into a browser URL."""
    if not path_value:
        return None

    candidate = Path(path_value)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

    try:
        relative_path = resolved.relative_to(data_root.resolve())
    except ValueError:
        return None

    return f"/media/{relative_path.as_posix()}"


def attach_result_urls(result: dict) -> dict:
    """Attach browser-safe URLs for generated artifacts."""
    enriched = dict(result)
    for field_name in (
        "final_song_path",
        "original_song_path",
        "vocal_path",
        "instrumental_path",
    ):
        media_url = build_media_url(enriched.get(field_name))
        if media_url:
            enriched[field_name.replace("_path", "_url")] = media_url
    return enriched


def build_rvc_models_payload() -> dict:
    """Expose official bundled RVC example voices to the frontend."""
    discovery = discover_runtime_models(project_root / "models" / "RVC1006Nvidia")
    models = []
    default_model_name = None

    for info in discovery["valid_models"]:
        model_name = Path(info["model_path"]).stem
        if info.get("is_default"):
            default_model_name = model_name
        models.append(
            {
                "name": model_name,
                "is_default": bool(info.get("is_default")),
                "model_path": info["model_path"],
                "index_path": info.get("index_path"),
            }
        )

    return {
        "models": models,
        "default_model_name": default_model_name,
        "count": len(models),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    html_path = project_root / "frontend" / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/v1/audio/upload")
async def upload_audio(file: UploadFile = File(...), audio_type: str = "mixed"):
    del audio_type

    audio_id = f"aud_{uuid4().hex[:8]}"
    upload_path = get_upload_path(audio_id)
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"audio_id": audio_id, "status": "uploaded"}


@app.get("/api/v1/style/recommend/{audio_id}")
async def recommend_style(audio_id: str):
    pipeline = require_pipeline()
    audio_path = get_upload_path(audio_id)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到上传音频: {audio_id}")

    pipeline.preprocessor.process(str(audio_path))
    try:
        voice_profile = pipeline.voice_cloner.train_or_encode(str(audio_path), audio_id, "quick")
    except Exception as exc:
        voice_profile = {
            "user_id": audio_id,
            "voice_model_path": None,
            "mode": "bypass",
            "quality_metrics": {},
            "error": str(exc),
        }

    recommendations = pipeline.style_recommender.recommend(str(audio_path), voice_profile)
    return {"recommendations": recommendations, "voice_profile": voice_profile}


@app.post("/api/v1/song/generate")
async def generate_song(request: GenerateRequest):
    pipeline = require_pipeline()
    audio_path = get_upload_path(request.audio_id)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到上传音频: {request.audio_id}")

    task_id = f"task_{uuid4().hex[:8]}"
    run_pipeline = partial(
        pipeline.run,
        audio_path=str(audio_path),
        user_id=request.audio_id,
        lyrics=request.lyrics,
        style=request.style,
        voice_model_name=request.rvc_model_name,
        voice_model_path=request.rvc_model_path,
    )
    result = await run_in_threadpool(run_pipeline)

    return {
        "task_id": task_id,
        "status": "completed",
        "result": attach_result_urls(result),
    }


@app.get("/api/v1/rvc/models")
async def list_rvc_models():
    return build_rvc_models_payload()


@app.get("/api/v1/pipeline/status")
async def pipeline_status():
    try:
        pipeline = get_pipeline()
    except RuntimeError as exc:
        return {
            "available": False,
            "error": str(exc),
            "pipeline_initialized": False,
        }

    status = pipeline.get_status()
    status["available"] = True
    status["error"] = None
    status["pipeline_initialized"] = True
    return status


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_initialized": _pipeline is not None,
        "pipeline_error": _pipeline_error,
    }
