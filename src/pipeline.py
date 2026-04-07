"""
完整Pipeline编排器 - 方案A重构版
架构: ACE-Step生成完整演唱歌曲 -> 人声分离 -> SVC音色转换 -> 重新混音
"""
from pathlib import Path
from typing import Dict, Optional
import json

from src.preprocessing.audio_cleaner import AudioPreprocessor
from src.voice_cloning.rvc_trainer import VoiceCloner
from src.music_generation.music_generator_svc import MusicGeneratorSVC
from src.style_recommendation.recommendation_engine import StyleRecommender
from src.postprocessing.audio_mixer import AudioMixer

class MusicAIPipeline:
    """AI音乐生成Pipeline - 方案A架构"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 初始化各模块
        self.preprocessor = AudioPreprocessor()
        self.voice_cloner = VoiceCloner(mode="rvc")  # 使用RVC进行声音克隆
        self.music_generator = MusicGeneratorSVC()  # 使用新的SVC架构音乐生成器
        self.style_recommender = StyleRecommender()
        self.mixer = AudioMixer()

        # 工作目录 - 使用绝对路径
        self.project_root = Path(__file__).parent.parent
        self.work_dir = self.project_root / "data"
        self.work_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 60)
        print("Pipeline初始化完成（方案A架构）")
        print("=" * 60)
        print(f"  架构: ACE-Step -> 人声分离 -> SVC转换 -> 重新混音")
        print(f"  采样率: 48000 Hz")
        print(f"  工作目录: {self.work_dir}")
        print("=" * 60 + "\n")

    def run(
        self,
        audio_path: str,
        user_id: str,
        lyrics: str = None,
        style: str = None,
        voice_model_name: str = None,
        voice_model_path: str = None,
    ) -> Dict:
        """
        完整流程 - 方案A架构

        流程:
        1. 音频预处理
        2. 声音克隆（训练RVC模型）
        3. 风格推荐（基于音频特征）
        4. 音乐生成（ACE-Step -> 人声分离 -> SVC转换 -> 重新混音）
        5. 返回结果

        输入:
            audio_path: 用户上传的音频 (说话/清唱)
            user_id: 用户ID
            lyrics: 歌词 (可选, 不提供则自动生成)
            style: 风格 (可选, 不提供则推荐)
            voice_model_name: 指定官方RVC示例音色名（可选）
            voice_model_path: 指定真实RVC模型路径（可选）

        输出:
            {
                "final_song_path": "...",  # 最终歌曲
                "original_song_path": "...",  # ACE-Step原始输出
                "vocal_path": "...",  # 转换后的人声
                "instrumental_path": "...",  # 伴奏
                "recommendations": [...],  # 风格推荐
                "metadata": {...}
            }
        """
        print(f"\n[Pipeline] 开始处理用户 {user_id} 的音频...")
        print(f"  输入音频: {audio_path}")

        # 为每个用户创建独立的输出文件夹
        user_output_dir = self.work_dir / "outputs" / user_id
        user_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {user_output_dir}")

        # ===== 模块1: 音频预处理 =====
        print("\n[1/5] 音频预处理...")
        clean_audio, audio_metadata = self.preprocessor.process(
            audio_path, audio_type="mixed"
        )
        clean_audio_path = user_output_dir / f"{user_id}_clean.wav"
        self.preprocessor.save(clean_audio, str(clean_audio_path))
        print(f"  ✓ 预处理完成: {clean_audio_path}")

        # ===== 模块2: 声音克隆 =====
        print("\n[2/5] 声音克隆...")
        try:
            voice_profile = self.voice_cloner.train_or_encode(
                str(clean_audio_path),
                user_id,
                training_mode="quick",
                preferred_model_name=voice_model_name,
                preferred_model_path=voice_model_path,
            )
            print(f"  ✓ 声音配置已解析: {voice_profile.get('voice_model_path', 'N/A')}")
        except Exception as e:
            voice_profile = {
                "user_id": user_id,
                "voice_model_path": None,
                "mode": "bypass",
                "quality_metrics": {},
                "error": str(e)
            }
            print(f"  ⚠ {e}")
            print("  保留ACE-Step原始人声")

        # ===== 模块3: 风格推荐 =====
        print("\n[3/5] 风格推荐...")
        recommendations = self.style_recommender.recommend(
            str(clean_audio_path), voice_profile, top_k=3
        )

        # 如果用户未指定风格,使用推荐的第一个
        if not style:
            style = recommendations[0]["style"]
            print(f"  ✓ 推荐风格: {style} (置信度: {recommendations[0]['confidence']:.2f})")
        else:
            print(f"  ✓ 使用指定风格: {style}")

        # 如果用户未提供歌词,生成简单示例
        if not lyrics:
            lyrics = self._generate_sample_lyrics(style)
            print(f"  ✓ 生成示例歌词")

        # ===== 模块4: 音乐生成（方案A架构）=====
        print("\n[4/5] 音乐生成（方案A架构）...")
        print(f"  流程: ACE-Step -> 人声分离 -> SVC转换 -> 重新混音")

        generation_params = {
            "tempo": 80,
            "key": "C_major",
            "duration": 90,  # 1分30秒
            "language": "zh"
        }

        music_result = self.music_generator.generate(
            lyrics, style, voice_profile, generation_params
        )

        print(f"  ✓ 音乐生成完成")
        print(f"    - 原始歌曲: {music_result['original_song_path']}")
        print(f"    - 转换人声: {music_result['vocal_path']}")
        print(f"    - 伴奏: {music_result['instrumental_path']}")
        print(f"    - 最终歌曲: {music_result['final_song_path']}")

        # ===== 模块5: 结果整合 =====
        print("\n[5/5] 整合结果...")

        result = {
            "final_song_path": str(music_result["final_song_path"]),
            "original_song_path": str(music_result["original_song_path"]),
            "vocal_path": str(music_result["vocal_path"]),
            "instrumental_path": str(music_result["instrumental_path"]),
            "recommendations": recommendations,
            "selected_style": style,
            "lyrics": lyrics,
            "metadata": {
                "audio_quality": audio_metadata,
                "voice_profile": voice_profile,
                "requested_voice_model_name": voice_model_name,
                "requested_voice_model_path": voice_model_path,
                "generation_params": generation_params,
                "architecture": "方案A: ACE-Step + SVC",
                "model_status": music_result["metadata"]["model_status"]
            }
        }

        # 保存结果元数据
        result_json_path = user_output_dir / "metadata.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 元数据已保存: {result_json_path}")

        # ===== 完成 =====
        print(f"\n[Pipeline] 完成! ")
        print(f"  输出目录: {user_output_dir}")
        print(f"  最终歌曲: {music_result['final_song_path']}")
        print(f"  架构: {result['metadata']['architecture']}")

        return result

    def _generate_sample_lyrics(self, style: str) -> str:
        """生成示例歌词"""
        sample_lyrics = {
            "pop_ballad": "夜空中最亮的星\n能否听清\n那仰望的人\n心底的孤独和叹息",
            "folk_acoustic": "走在乡间的小路上\n暮归的老牛是我同伴\n蓝天配朵夕阳在胸膛",
            "r&b_soul": "说散就散\n我的眼泪都掉了\n你说的话\n我都记得",
            "rock": "海阔天空\n在勇敢以后\n要拿执着\n将命运的锁打破",
            "electronic": "霓虹灯闪烁\n节奏在跳动\n让我们一起\n追逐梦想",
            "jazz": "月光洒在窗台\n旋律在流淌\n这一刻\n时间静止",
            "classical": "山川河流\n日月星辰\n永恒的美\n在你心中"
        }
        return sample_lyrics.get(style, "这是一首美妙的歌\n唱出心中的快乐")

    def get_status(self) -> Dict:
        """获取Pipeline状态"""
        return {
            "architecture": "方案A: ACE-Step + SVC",
            "modules": {
                "preprocessor": self.preprocessor is not None,
                "voice_cloner": self.voice_cloner is not None,
                "music_generator": self.music_generator is not None,
                "style_recommender": self.style_recommender is not None,
                "mixer": self.mixer is not None
            },
            "model_status": {
                "ace_step": self.music_generator.ace_step is not None and self.music_generator.ace_step.is_ready(),
                "ace_step_backend": (
                    getattr(self.music_generator.ace_step, "official_backend_name", "local_wrapper")
                    if self.music_generator.ace_step is not None
                    else None
                ),
                "demucs": self.music_generator.vocal_separator is not None and self.music_generator.vocal_separator.is_ready(),
                "rvc": self.music_generator.svc_converter is not None and self.music_generator.svc_converter.is_ready()
            }
        }
