"""
项目路径统一配置
所有模型和数据路径都在项目根目录下的models和data目录
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
ACE_STEP_DIR = MODELS_DIR / "ace_step"
FISH_SPEECH_DIR = MODELS_DIR / "fish_speech"
RVC_DIR = MODELS_DIR / "rvc_pretrained"
USER_VOICES_DIR = MODELS_DIR / "user_voices"

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# 确保目录存在
def ensure_dirs():
    """创建所有必需的目录"""
    for dir_path in [
        MODELS_DIR, ACE_STEP_DIR, FISH_SPEECH_DIR, RVC_DIR, USER_VOICES_DIR,
        DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, OUTPUTS_DIR
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

# 获取路径函数
def get_model_path(model_name: str) -> Path:
    """获取模型路径"""
    paths = {
        "ace_step": ACE_STEP_DIR,
        "fish_speech": FISH_SPEECH_DIR,
        "rvc": RVC_DIR,
        "user_voices": USER_VOICES_DIR
    }
    return paths.get(model_name, MODELS_DIR / model_name)

def get_user_model_path(user_id: str, model_type: str = "rvc") -> Path:
    """获取用户模型路径"""
    if model_type == "rvc":
        return USER_VOICES_DIR / f"user_{user_id}_voice.pth"
    elif model_type == "fish_speech":
        return USER_VOICES_DIR / f"user_{user_id}_prompt.pt"
    return USER_VOICES_DIR / f"user_{user_id}.pt"

def get_output_path(filename: str) -> Path:
    """获取输出文件路径"""
    return OUTPUTS_DIR / filename
