"""API route helper tests."""
from __future__ import annotations

from pathlib import Path

from src.api import routes


def test_build_media_url_and_attach_result_urls(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_file = data_root / "outputs" / "song.wav"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"wav")

    monkeypatch.setattr(routes, "project_root", tmp_path)
    monkeypatch.setattr(routes, "data_root", data_root)

    media_url = routes.build_media_url(str(output_file))
    assert media_url == "/media/outputs/song.wav"

    enriched = routes.attach_result_urls(
        {
            "final_song_path": str(output_file),
            "original_song_path": str(output_file),
            "vocal_path": str(output_file),
            "instrumental_path": str(output_file),
        }
    )

    assert enriched["final_song_url"] == "/media/outputs/song.wav"
    assert enriched["original_song_url"] == "/media/outputs/song.wav"


def test_build_rvc_models_payload(monkeypatch):
    monkeypatch.setattr(
        routes,
        "discover_runtime_models",
        lambda repo_path: {
            "valid_models": [
                {
                    "model_path": "/runtime/kikiV1.pth",
                    "index_path": "/runtime/kikiV1.index",
                    "is_default": True,
                },
                {
                    "model_path": "/runtime/keruanV1.pth",
                    "index_path": None,
                    "is_default": False,
                },
            ]
        },
    )

    payload = routes.build_rvc_models_payload()

    assert payload["default_model_name"] == "kikiV1"
    assert payload["count"] == 2
    assert payload["models"][0]["name"] == "kikiV1"
