"""
模型配置和路径管理
"""
from pathlib import Path


class ModelConfig:
    """项目中各类模型的路径配置。"""

    BASE_DIR = Path(__file__).parent.parent.parent
    MODELS_DIR = BASE_DIR / "models"

    ACE_STEP = {
        "model_path": MODELS_DIR / "Ace-Step1.5",
        "repo_id": "ACE-Step/Ace-Step1.5",
        "description": "ACE-Step 本地音乐生成模型",
    }

    DEMUCS = {
        "model_path": MODELS_DIR / "demucs",
        "description": "Demucs 本地人声分离权重",
    }

    RVC = {
        "model_path": MODELS_DIR / "RVC1006Nvidia",
        "pretrained_path": MODELS_DIR / "rvc_pretrained",
        "user_model_path": MODELS_DIR / "user_voices",
        "description": "RVC 音色转换模型与用户音色目录",
    }

    USER_VOICES_DIR = MODELS_DIR / "user_voices"

    ACTIVE_MODELS = {
        "music_generation": "ace_step",
        "voice_clone": "rvc",
        "voice_conversion": "rvc",
        "vocal_separation": "demucs",
    }

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        model_map = {
            "ace_step": cls.ACE_STEP["model_path"],
            "demucs": cls.DEMUCS["model_path"],
            "rvc": cls.RVC["model_path"],
            "rvc_pretrained": cls.RVC["pretrained_path"],
            "user_voices": cls.RVC["user_model_path"],
        }
        return model_map.get(model_name, cls.MODELS_DIR / model_name)

    @classmethod
    def check_models_exist(cls) -> dict:
        return {
            "ace_step": cls.ACE_STEP["model_path"].exists(),
            "demucs": cls.DEMUCS["model_path"].exists(),
            "rvc": cls.RVC["model_path"].exists(),
            "rvc_pretrained": cls.RVC["pretrained_path"].exists(),
        }

    @classmethod
    def get_missing_models(cls) -> list:
        status = cls.check_models_exist()
        return [name for name, exists in status.items() if not exists]
