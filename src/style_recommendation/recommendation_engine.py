"""
模块4: 风格推荐 - 改进版
功能: 音频特征提取、风格分类、推荐生成
改进: 使用更全面的特征提取和多维度评分机制
"""
import librosa
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StyleRecommender:
    """风格推荐器 - 基于真实音频特征分析"""

    def __init__(self):
        self.style_database = self._load_style_database()
        self.style_profiles = self._load_style_profiles()

    def recommend(self, audio_path: str, voice_profile: Dict,
                  top_k: int = 3) -> List[Dict]:
        """
        基于音频特征推荐音乐风格

        Args:
            audio_path: 音频路径
            voice_profile: 声音档案
            top_k: 返回前K个推荐

        Returns:
            推荐列表
        """
        print(f"\n[风格推荐] 分析音频特征...")
        print(f"  音频路径: {audio_path}")

        # 1. 提取全面特征
        features = self._extract_features(audio_path)

        # 2. 计算风格匹配度
        scores = self._match_styles(features)

        # 3. 生成推荐
        recommendations = self._generate_recommendations(scores, features, top_k)

        # 打印特征摘要
        self._print_feature_summary(features, recommendations)

        return recommendations

    def _extract_features(self, audio_path: str) -> Dict:
        """
        提取全面的音频特征

        特征包括:
        - 音高特征: f0_mean, f0_std, f0_range, vocal_range_semitones
        - 音色特征: mfcc, spectral_centroid, spectral_bandwidth, spectral_rolloff
        - 动态特征: rms, zero_crossing_rate
        - 节奏特征: tempo
        - 和声特征: chroma
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=22050)

            # ===== 音高特征 =====
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]

            if len(f0_valid) > 0:
                f0_mean = float(np.mean(f0_valid))
                f0_std = float(np.std(f0_valid))
                f0_min = float(np.min(f0_valid))
                f0_max = float(np.max(f0_valid))
                vocal_range_semitones = float(12 * np.log2(f0_max / f0_min)) if f0_min > 0 else 0.0
                pitch_stability = float(1.0 - (f0_std / f0_mean)) if f0_mean > 0 else 0.0
            else:
                f0_mean = f0_std = f0_min = f0_max = 0.0
                vocal_range_semitones = 0.0
                pitch_stability = 0.0

            # ===== 音色特征 =====
            # MFCC (13维)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1).tolist()
            mfcc_std = mfcc.std(axis=1).tolist()

            # 频谱特征
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = float(np.mean(spectral_centroid))
            spectral_centroid_std = float(np.std(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = float(np.mean(spectral_rolloff))

            # 色度特征 (和声信息)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1).tolist()

            # ===== 动态特征 =====
            rms = librosa.feature.rms(y=y)
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))

            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            zcr_mean = float(np.mean(zero_crossing_rate))

            # ===== 节奏特征 =====
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # ===== 附加特征 =====
            # 亮度 (高频能量占比)
            brightness = float(np.mean(spectral_centroid) / sr)

            # 粗糙度 (频谱复杂度)
            roughness = float(np.std(spectral_bandwidth))

            return {
                # 音高特征
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_range": [f0_min, f0_max],
                "vocal_range_semitones": vocal_range_semitones,
                "pitch_stability": pitch_stability,

                # 音色特征
                "mfcc_mean": mfcc_mean,
                "mfcc_std": mfcc_std,
                "spectral_centroid_mean": spectral_centroid_mean,
                "spectral_centroid_std": spectral_centroid_std,
                "spectral_bandwidth_mean": spectral_bandwidth_mean,
                "spectral_rolloff_mean": spectral_rolloff_mean,
                "chroma_mean": chroma_mean,
                "brightness": brightness,
                "roughness": roughness,

                # 动态特征
                "loudness_mean": rms_mean,
                "loudness_std": rms_std,
                "zero_crossing_rate_mean": zcr_mean,

                # 节奏特征
                "tempo": float(tempo),

                # 元数据
                "duration": float(len(y) / sr),
                "sample_rate": sr
            }

        except Exception as e:
            print(f"  ⚠ 特征提取失败: {e}")
            return self._get_default_features()

    def _get_default_features(self) -> Dict:
        """返回默认特征值"""
        return {
            "f0_mean": 200.0,
            "f0_std": 50.0,
            "f0_range": [150.0, 300.0],
            "vocal_range_semitones": 12.0,
            "pitch_stability": 0.75,
            "mfcc_mean": [0.0] * 13,
            "mfcc_std": [1.0] * 13,
            "spectral_centroid_mean": 2000.0,
            "spectral_centroid_std": 500.0,
            "spectral_bandwidth_mean": 1500.0,
            "spectral_rolloff_mean": 4000.0,
            "chroma_mean": [0.1] * 12,
            "brightness": 0.1,
            "roughness": 500.0,
            "loudness_mean": 0.3,
            "loudness_std": 0.1,
            "zero_crossing_rate_mean": 0.1,
            "tempo": 120.0,
            "duration": 10.0,
            "sample_rate": 22050
        }

    def _match_styles(self, features: Dict) -> Dict[str, float]:
        """
        基于多维度特征匹配风格

        评分机制:
        - 每个风格有特定的特征期望
        - 计算实际特征与期望特征的匹配度
        - 综合多个维度给出最终分数
        """
        scores = {}

        # 获取风格特征期望
        style_expectations = self.style_profiles

        # 计算每个风格的匹配度
        for style_name, expectations in style_expectations.items():
            score = self._calculate_style_score(features, expectations)
            scores[style_name] = max(0.0, min(1.0, score))  # 限制在0-1之间

        return scores

    def _calculate_style_score(self, features: Dict, expectations: Dict) -> float:
        """计算单个风格的匹配分数"""
        total_score = 0.0
        total_weight = 0.0

        # 1. 音高维度 (权重: 0.25)
        if "pitch" in expectations:
            pitch_score = self._score_pitch_dimension(features, expectations["pitch"])
            total_score += pitch_score * 0.25
            total_weight += 0.25

        # 2. 音色维度 (权重: 0.30)
        if "timbre" in expectations:
            timbre_score = self._score_timbre_dimension(features, expectations["timbre"])
            total_score += timbre_score * 0.30
            total_weight += 0.30

        # 3. 动态维度 (权重: 0.25)
        if "dynamic" in expectations:
            dynamic_score = self._score_dynamic_dimension(features, expectations["dynamic"])
            total_score += dynamic_score * 0.25
            total_weight += 0.25

        # 4. 节奏维度 (权重: 0.20)
        if "rhythm" in expectations:
            rhythm_score = self._score_rhythm_dimension(features, expectations["rhythm"])
            total_score += rhythm_score * 0.20
            total_weight += 0.20

        # 归一化
        if total_weight > 0:
            return total_score / total_weight
        return 0.5  # 默认分数

    def _score_pitch_dimension(self, features: Dict, pitch_expect: Dict) -> float:
        """评分音高维度"""
        score = 0.0

        # 音高范围
        if "f0_mean_range" in pitch_expect:
            f0_mean = features["f0_mean"]
            min_f0, max_f0 = pitch_expect["f0_mean_range"]
            if min_f0 <= f0_mean <= max_f0:
                score += 0.4
            else:
                # 部分匹配
                distance = min(abs(f0_mean - min_f0), abs(f0_mean - max_f0))
                score += max(0.0, 0.4 - distance / 500.0)

        # 音高稳定性
        if "min_pitch_stability" in pitch_expect:
            stability = features["pitch_stability"]
            min_stability = pitch_expect["min_pitch_stability"]
            if stability >= min_stability:
                score += 0.3
            else:
                score += 0.3 * (stability / min_stability)

        # 音域
        if "max_vocal_range" in pitch_expect:
            vocal_range = features["vocal_range_semitones"]
            max_range = pitch_expect["max_vocal_range"]
            if vocal_range <= max_range:
                score += 0.3
            else:
                score += 0.3 * (max_range / vocal_range)

        return score

    def _score_timbre_dimension(self, features: Dict, timbre_expect: Dict) -> float:
        """评分音色维度"""
        score = 0.0

        # 频谱质心 (亮度)
        if "spectral_centroid_range" in timbre_expect:
            centroid = features["spectral_centroid_mean"]
            min_centroid, max_centroid = timbre_expect["spectral_centroid_range"]
            if min_centroid <= centroid <= max_centroid:
                score += 0.4
            else:
                distance = min(abs(centroid - min_centroid), abs(centroid - max_centroid))
                score += max(0.0, 0.4 - distance / 2000.0)

        # 频谱带宽
        if "spectral_bandwidth_range" in timbre_expect:
            bandwidth = features["spectral_bandwidth_mean"]
            min_bw, max_bw = timbre_expect["spectral_bandwidth_range"]
            if min_bw <= bandwidth <= max_bw:
                score += 0.3
            else:
                distance = min(abs(bandwidth - min_bw), abs(bandwidth - max_bw))
                score += max(0.0, 0.3 - distance / 1500.0)

        # 亮度
        if "brightness_range" in timbre_expect:
            brightness = features["brightness"]
            min_bright, max_bright = timbre_expect["brightness_range"]
            if min_bright <= brightness <= max_bright:
                score += 0.3
            else:
                distance = min(abs(brightness - min_bright), abs(brightness - max_bright))
                score += max(0.0, 0.3 - distance / 0.1)

        return score

    def _score_dynamic_dimension(self, features: Dict, dynamic_expect: Dict) -> float:
        """评分动态维度"""
        score = 0.0

        # 响度
        if "loudness_range" in dynamic_expect:
            loudness = features["loudness_mean"]
            min_loud, max_loud = dynamic_expect["loudness_range"]
            if min_loud <= loudness <= max_loud:
                score += 0.5
            else:
                distance = min(abs(loudness - min_loud), abs(loudness - max_loud))
                score += max(0.0, 0.5 - distance / 0.3)

        # 动态范围
        if "min_dynamic_range" in dynamic_expect:
            dynamic_range = features["loudness_std"]
            min_range = dynamic_expect["min_dynamic_range"]
            if dynamic_range >= min_range:
                score += 0.5
            else:
                score += 0.5 * (dynamic_range / min_range)

        return score

    def _score_rhythm_dimension(self, features: Dict, rhythm_expect: Dict) -> float:
        """评分节奏维度"""
        score = 0.0

        # 节奏
        if "tempo_range" in rhythm_expect:
            tempo = features["tempo"]
            min_tempo, max_tempo = rhythm_expect["tempo_range"]
            if min_tempo <= tempo <= max_tempo:
                score += 1.0
            else:
                distance = min(abs(tempo - min_tempo), abs(tempo - max_tempo))
                score += max(0.0, 1.0 - distance / 60.0)

        return score

    def _generate_recommendations(self, scores: Dict, features: Dict,
                                  top_k: int) -> List[Dict]:
        """生成推荐列表"""
        # 按分数排序
        sorted_styles = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, (style, confidence) in enumerate(sorted_styles[:top_k]):
            # 确保至少有一些推荐
            if confidence < 0.3 and i == 0:
                confidence = 0.5  # 最低推荐分数

            recommendations.append({
                "style": style,
                "confidence": round(confidence, 3),
                "reason": self._generate_reason(style, features),
                "example_songs": self.style_database.get(style, {}).get("examples", []),
                "features_matched": self._get_matched_features(style, features)
            })

        return recommendations

    def _generate_reason(self, style: str, features: Dict) -> str:
        """生成推荐理由"""
        f0_mean = features["f0_mean"]
        f0_range = features["f0_range"]
        vocal_range = features["vocal_range_semitones"]
        stability = features["pitch_stability"]
        brightness = features["brightness"]
        loudness = features["loudness_mean"]

        reasons = {
            "pop_ballad": (
                f"音域适中 ({f0_range[0]:.0f}-{f0_range[1]:.0f}Hz)，"
                f"音色温暖 (亮度{brightness:.2f})，"
                f"音高稳定 ({stability:.2f})，适合抒情表达"
            ),
            "folk_acoustic": (
                f"声音自然质朴，音域不过宽 ({vocal_range:.1f}半音)，"
                f"音色清澈，适合讲故事般的演唱"
            ),
            "r&b_soul": (
                f"音域较宽 ({vocal_range:.1f}半音)，"
                f"音色富有表现力，情感变化丰富，适合灵魂乐风格"
            ),
            "rock": (
                f"能量充沛 (响度{loudness:.2f})，"
                f"声音有力，适合高强度、激情的演唱"
            ),
            "electronic": (
                f"音色明亮 (亮度{brightness:.2f})，"
                f"节奏感强，适合电子音乐风格"
            ),
            "jazz": (
                f"音高变化丰富，"
                f"音色细腻，适合即兴和复杂的音乐表达"
            ),
            "classical": (
                f"音域宽广，音色纯净，"
                f"技巧要求高，适合古典音乐风格"
            )
        }

        return reasons.get(style, "适合该风格")

    def _get_matched_features(self, style: str, features: Dict) -> Dict:
        """获取匹配的特征摘要"""
        return {
            "f0_mean": round(features["f0_mean"], 1),
            "vocal_range_semitones": round(features["vocal_range_semitones"], 1),
            "pitch_stability": round(features["pitch_stability"], 2),
            "brightness": round(features["brightness"], 3),
            "loudness": round(features["loudness_mean"], 3),
            "tempo": round(features["tempo"], 1)
        }

    def _print_feature_summary(self, features: Dict, recommendations: List[Dict]):
        """打印特征摘要"""
        print(f"\n  音频特征摘要:")
        print(f"    音高: {features['f0_mean']:.1f}Hz (范围: {features['f0_range'][0]:.0f}-{features['f0_range'][1]:.0f}Hz)")
        print(f"    音域: {features['vocal_range_semitones']:.1f} 半音")
        print(f"    稳定性: {features['pitch_stability']:.2f}")
        print(f"    亮度: {features['brightness']:.3f}")
        print(f"    响度: {features['loudness_mean']:.3f}")
        print(f"    节奏: {features['tempo']:.1f} BPM")

        print(f"\n  推荐结果:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec['style']} (置信度: {rec['confidence']:.2f})")
            print(f"       理由: {rec['reason']}")

    def _load_style_database(self) -> Dict:
        """加载风格数据库"""
        return {
            "pop_ballad": {
                "description": "流行抒情",
                "examples": ["告白气球", "说好不哭", "晴天", "后来"],
                "characteristics": "旋律优美，情感丰富，适合大众传唱"
            },
            "folk_acoustic": {
                "description": "民谣",
                "examples": ["成都", "南山南", "董小姐", "安和桥"],
                "characteristics": "简单自然，贴近生活，富有故事性"
            },
            "r&b_soul": {
                "description": "R&B灵魂乐",
                "examples": ["说散就散", "飘向北方", "演员", "体面"],
                "characteristics": "节奏感强，情感细腻，技巧丰富"
            },
            "rock": {
                "description": "摇滚",
                "examples": ["海阔天空", "光辉岁月", "真的爱你", "蓝莲花"],
                "characteristics": "激情澎湃，力量感强，个性鲜明"
            },
            "electronic": {
                "description": "电子音乐",
                "examples": ["江南Style", "Uptown Funk", "Blinding Lights"],
                "characteristics": "节奏强烈，音色丰富，时尚前卫"
            },
            "jazz": {
                "description": "爵士乐",
                "examples": ["Fly Me to the Moon", "Autumn Leaves", "Summertime"],
                "characteristics": "即兴自由，和声复杂，格调高雅"
            },
            "classical": {
                "description": "古典音乐",
                "examples": ["卡农", "月光奏鸣曲", "四季"],
                "characteristics": "结构严谨，技巧精湛，艺术性强"
            }
        }

    def _load_style_profiles(self) -> Dict:
        """
        加载风格特征期望

        每个风格定义了对音频特征的期望范围
        """
        return {
            "pop_ballad": {
                "pitch": {
                    "f0_mean_range": (150, 300),
                    "min_pitch_stability": 0.65,
                    "max_vocal_range": 18
                },
                "timbre": {
                    "spectral_centroid_range": (1500, 3000),
                    "spectral_bandwidth_range": (1000, 2000),
                    "brightness_range": (0.07, 0.15)
                },
                "dynamic": {
                    "loudness_range": (0.2, 0.5),
                    "min_dynamic_range": 0.05
                },
                "rhythm": {
                    "tempo_range": (60, 90)
                }
            },
            "folk_acoustic": {
                "pitch": {
                    "f0_mean_range": (130, 250),
                    "min_pitch_stability": 0.60,
                    "max_vocal_range": 15
                },
                "timbre": {
                    "spectral_centroid_range": (1200, 2500),
                    "spectral_bandwidth_range": (800, 1500),
                    "brightness_range": (0.05, 0.12)
                },
                "dynamic": {
                    "loudness_range": (0.15, 0.4),
                    "min_dynamic_range": 0.04
                },
                "rhythm": {
                    "tempo_range": (70, 100)
                }
            },
            "r&b_soul": {
                "pitch": {
                    "f0_mean_range": (180, 350),
                    "min_pitch_stability": 0.55,
                    "max_vocal_range": 25
                },
                "timbre": {
                    "spectral_centroid_range": (2000, 3500),
                    "spectral_bandwidth_range": (1500, 2500),
                    "brightness_range": (0.10, 0.18)
                },
                "dynamic": {
                    "loudness_range": (0.25, 0.55),
                    "min_dynamic_range": 0.08
                },
                "rhythm": {
                    "tempo_range": (70, 100)
                }
            },
            "rock": {
                "pitch": {
                    "f0_mean_range": (200, 400),
                    "min_pitch_stability": 0.50,
                    "max_vocal_range": 30
                },
                "timbre": {
                    "spectral_centroid_range": (2500, 4500),
                    "spectral_bandwidth_range": (2000, 3500),
                    "brightness_range": (0.12, 0.25)
                },
                "dynamic": {
                    "loudness_range": (0.4, 0.7),
                    "min_dynamic_range": 0.1
                },
                "rhythm": {
                    "tempo_range": (100, 140)
                }
            },
            "electronic": {
                "pitch": {
                    "f0_mean_range": (150, 350),
                    "min_pitch_stability": 0.55,
                    "max_vocal_range": 20
                },
                "timbre": {
                    "spectral_centroid_range": (2500, 4000),
                    "spectral_bandwidth_range": (2000, 3000),
                    "brightness_range": (0.15, 0.25)
                },
                "dynamic": {
                    "loudness_range": (0.3, 0.6),
                    "min_dynamic_range": 0.08
                },
                "rhythm": {
                    "tempo_range": (110, 140)
                }
            },
            "jazz": {
                "pitch": {
                    "f0_mean_range": (150, 300),
                    "min_pitch_stability": 0.45,
                    "max_vocal_range": 28
                },
                "timbre": {
                    "spectral_centroid_range": (1800, 3200),
                    "spectral_bandwidth_range": (1200, 2200),
                    "brightness_range": (0.08, 0.16)
                },
                "dynamic": {
                    "loudness_range": (0.2, 0.5),
                    "min_dynamic_range": 0.1
                },
                "rhythm": {
                    "tempo_range": (80, 130)
                }
            },
            "classical": {
                "pitch": {
                    "f0_mean_range": (180, 350),
                    "min_pitch_stability": 0.70,
                    "max_vocal_range": 35
                },
                "timbre": {
                    "spectral_centroid_range": (2000, 3500),
                    "spectral_bandwidth_range": (1500, 2500),
                    "brightness_range": (0.09, 0.17)
                },
                "dynamic": {
                    "loudness_range": (0.25, 0.55),
                    "min_dynamic_range": 0.12
                },
                "rhythm": {
                    "tempo_range": (60, 120)
                }
            }
        }
