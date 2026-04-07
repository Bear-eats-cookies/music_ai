"""Pipeline tests."""
from __future__ import annotations

import json
from pathlib import Path

import src.pipeline as pipeline_module


def test_pipeline_run_passes_requested_rvc_model(tmp_path, monkeypatch):
    class DummyPreprocessor:
        def process(self, audio_path, audio_type="mixed"):
            return [0.0] * 16, {"quality_score": 0.9, "audio_type": audio_type}

        def save(self, audio, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"clean")

    class DummyVoiceCloner:
        def __init__(self, mode="rvc"):
            self.mode = mode
            self.calls = []

        def train_or_encode(
            self,
            audio_path,
            user_id,
            training_mode="quick",
            preferred_model_name=None,
            preferred_model_path=None,
        ):
            self.calls.append(
                {
                    "audio_path": audio_path,
                    "user_id": user_id,
                    "training_mode": training_mode,
                    "preferred_model_name": preferred_model_name,
                    "preferred_model_path": preferred_model_path,
                }
            )
            return {
                "user_id": user_id,
                "voice_model_path": preferred_model_path or f"runtime/{preferred_model_name}.pth",
                "index_path": None,
                "mode": "rvc_runtime_default",
                "quality_metrics": {},
                "source": "test_stub",
            }

    class DummyMusicGenerator:
        def __init__(self):
            self.ace_step = None
            self.vocal_separator = None
            self.svc_converter = None

        def generate(self, lyrics, style, voice_profile, params):
            outputs_dir = tmp_path / "generated"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            artifact_paths = {
                "original_song_path": outputs_dir / "original.wav",
                "vocal_path": outputs_dir / "vocal.wav",
                "instrumental_path": outputs_dir / "accompaniment.wav",
                "final_song_path": outputs_dir / "final.wav",
            }
            for artifact_path in artifact_paths.values():
                artifact_path.write_bytes(b"wav")
            return {
                "song_id": "song_test",
                **{key: str(value) for key, value in artifact_paths.items()},
                "metadata": {"model_status": {"ace_step": True, "demucs": True, "rvc": True}},
            }

    class DummyStyleRecommender:
        def recommend(self, audio_path, voice_profile, top_k=3):
            return [{"style": "rock", "confidence": 0.95, "reason": "test"}]

    class DummyMixer:
        pass

    monkeypatch.setattr(pipeline_module, "AudioPreprocessor", DummyPreprocessor)
    monkeypatch.setattr(pipeline_module, "VoiceCloner", DummyVoiceCloner)
    monkeypatch.setattr(pipeline_module, "MusicGeneratorSVC", DummyMusicGenerator)
    monkeypatch.setattr(pipeline_module, "StyleRecommender", DummyStyleRecommender)
    monkeypatch.setattr(pipeline_module, "AudioMixer", DummyMixer)

    pipeline = pipeline_module.MusicAIPipeline()
    pipeline.project_root = tmp_path
    pipeline.work_dir = tmp_path / "data"
    pipeline.work_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.run(
        audio_path=str(tmp_path / "input.wav"),
        user_id="aud_123",
        lyrics="test lyrics",
        style="rock",
        voice_model_name="kikiV1",
        voice_model_path=str(tmp_path / "preferred_model.pth"),
    )

    assert result["final_song_path"].endswith("final.wav")
    assert pipeline.voice_cloner.calls[0]["preferred_model_name"] == "kikiV1"
    assert pipeline.voice_cloner.calls[0]["preferred_model_path"].endswith("preferred_model.pth")

    clean_audio_path = tmp_path / "data" / "outputs" / "aud_123" / "aud_123_clean.wav"
    metadata_path = tmp_path / "data" / "outputs" / "aud_123" / "metadata.json"
    assert clean_audio_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata"]["requested_voice_model_name"] == "kikiV1"
    assert metadata["metadata"]["requested_voice_model_path"].endswith("preferred_model.pth")
