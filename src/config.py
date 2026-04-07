"""
项目配置文件 - 方案A重构版
"""
from pathlib import Path

class Config:
    # 项目路径
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"

    # 音频参数 - 统一使用48kHz
    SAMPLE_RATE = 48000  # 统一采样率：48kHz
    TARGET_DURATION = 180  # 目标时长：3分钟

    # 方案A架构配置
    ARCHITECTURE = "SVC"  # 'SVC' (方案A) 或 'SVS' (方案B)

    # 模型配置
    VOICE_CLONE_MODE = "rvc"  # 'rvc' (推荐) or 'fish_speech'
    MUSIC_GEN_MODEL = "ace_step"  # 'ace_step' or 'musicgen'
    VOCAL_SEPARATOR_MODEL = "demucs"  # 'demucs' (人声分离)
    SVC_MODEL = "rvc"  # 'rvc' or 'so_vits_svc' (音色转换)

    # ACE-Step配置
    ACE_STEP_MODEL_PATH = MODELS_DIR / "Ace-Step1.5"
    ACE_STEP_GENERATE_FULL_SONG = True  # 生成完整歌曲（含人声）

    # Demucs配置
    DEMUCS_MODEL_NAME = "htdemucs"  # 'htdemucs', 'htdemucs_ft', 'htdemucs_6s'

    # RVC配置
    RVC_MODEL_PATH = MODELS_DIR / "RVC1006Nvidia"
    RVC_TRAINING_MODE = "quick"  # 'quick', 'standard', 'high_quality'

    # 音频处理参数
    AUDIO_FORMAT = "wav"
    AUDIO_CHANNELS = 2  # 立体声
    AUDIO_BIT_DEPTH = 16

    # 混音参数
    VOCAL_VOLUME = 0.4  # 人声音量
    ACCOMPANIMENT_VOLUME = 0.6  # 伴奏音量
    MASTER_VOLUME = 0.95  # 主音量

    # GPU配置
    USE_GPU = True
    GPU_MEMORY_FRACTION = 0.8
    DEVICE = "cuda" if USE_GPU else "cpu"

    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # Celery配置
    CELERY_BROKER_URL = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/1"
    CELERY_TASK_TIMEOUT = 600  # 10分钟超时

    # Redis配置
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0

    # 数据库配置
    DATABASE_URL = "postgresql://user:pass@localhost/music_ai"

    # 风格推荐配置
    STYLE_RECOMMENDATION_TOP_K = 3
    MIN_RECOMMENDATION_CONFIDENCE = 0.3  # 最低推荐置信度

    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
