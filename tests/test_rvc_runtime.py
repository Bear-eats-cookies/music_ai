"""RVC runtime selection tests."""
from __future__ import annotations

import src.voice_cloning.rvc_runtime as runtime


def test_select_rvc_model_prefers_request_over_env(monkeypatch):
    def fake_inspect_user_model(model_path):
        return {
            "model_path": str(model_path),
            "valid": True,
            "index_path": None,
        }

    def fake_find_runtime_model_by_name(model_name, repo_path=None):
        if model_name == "kikiV1":
            return {
                "model_path": "/runtime/kikiV1.pth",
                "valid": True,
                "index_path": "/runtime/kikiV1.index",
            }
        return None

    monkeypatch.setattr(runtime, "inspect_user_model", fake_inspect_user_model)
    monkeypatch.setattr(runtime, "find_runtime_model_by_name", fake_find_runtime_model_by_name)
    monkeypatch.setenv("MUSIC_AI_RVC_MODEL_PATH", "/env/model.pth")
    monkeypatch.setenv("MUSIC_AI_RVC_MODEL_NAME", "envVoice")

    result = runtime.select_rvc_model(
        preferred_model_name="kikiV1",
        preferred_model_path="/preferred/model.pth",
        allow_runtime_default=True,
    )

    assert result["model_path"] == "/preferred/model.pth"
    assert result["source"] == "preferred_path"
