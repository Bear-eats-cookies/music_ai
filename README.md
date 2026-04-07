# music_ai

一个基于 `FastAPI + ACE-Step + Demucs + RVC` 的 AI 音乐生成项目。

项目当前的主链路是：

- 使用 `ACE-Step` 生成歌曲
- 使用 `Demucs` 做人声分离
- 使用 `RVC` 做音色转换 / 换声
- 提供前端页面、API 接口和完整链路测试脚本

## 核心能力

- 上传用户音频样本
- 基于音频做风格推荐
- 输入歌词并生成歌曲
- 使用官方 RVC 示例音色进行换声
- 返回可直接在浏览器播放的生成结果 URL
- 提供模型检查、RVC 示例测试和完整链路测试脚本

## 当前技术路线

默认启用的模型链路：

- 音乐生成：`ACE-Step`
- 人声分离：`Demucs`
- 音色转换：`RVC`

更详细的模型名称、目录和依赖说明见：

- [项目模型文档.md](./项目模型文档.md)

## 项目结构

```text
music_ai/
├─ frontend/                 前端页面与静态资源
├─ src/
│  ├─ api/                   FastAPI 路由
│  ├─ music_generation/      ACE-Step / 音乐生成逻辑
│  ├─ preprocessing/         音频预处理 / Demucs 分离
│  ├─ postprocessing/        混音与后处理
│  ├─ style_recommendation/  风格推荐
│  ├─ voice_cloning/         RVC 运行时与模型选择
│  └─ voice_conversion/      RVC 转换逻辑
├─ scripts/                  测试与辅助脚本
├─ tests/                    单元测试
├─ data/                     上传与输出数据
├─ models/                   本地模型目录（不会上传到 GitHub）
├─ run.py                    推荐启动方式
├─ main.py                   备用启动方式
└─ verify_models.py          模型检查脚本
```

## 环境要求

- Python `3.10.13`
- 建议使用 Windows + NVIDIA GPU
- 建议提前准备好本地模型目录 `models/`

主要依赖见：

- `requirements.txt`

## 模型准备

本项目仓库不会上传模型文件，模型需要你在本地单独准备。

默认相关目录：

- `models/Ace-Step1.5`
- `models/demucs`
- `models/RVC1006Nvidia`
- `models/rvc_pretrained`
- `models/user_voices`

如果你只是先验证链路，优先准备：

- `ACE-Step` 主模型及其子模型
- `Demucs` 权重
- `RVC1006Nvidia` 运行时
- RVC 官方示例音色，如 `kikiV1`

详细模型名和文件名请看：

- [项目模型文档.md](./项目模型文档.md)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你要优先走官方 ACE-Step 推理链路，还需要额外安装官方 `acestep` 包。

### 2. 检查模型

```bash
python verify_models.py
```

### 3. 启动服务

推荐使用：

```bash
python run.py
```

默认访问地址：

- 首页：`http://localhost:8300`
- API 文档：`http://localhost:8300/docs`

备用启动方式：

```bash
python main.py
```

备用端口：

- 首页：`http://localhost:8080`
- API 文档：`http://localhost:8080/docs`

## 常用脚本

检查官方 `acestep` 是否已安装：

```bash
python -c "import importlib.util; print(importlib.util.find_spec('acestep') is not None)"
```

验证模型：

```bash
python verify_models.py
```

测试官方 RVC 示例音色：

```bash
python scripts\test_rvc_example.py --output data\outputs\rvc_example_test.wav
```

跑完整链路示例：

```bash
python scripts\test_full_pipeline_example.py --rvc-model-name kikiV1
```

## API 概览

当前主要接口：

- `GET /`：前端页面
- `POST /api/v1/audio/upload`：上传音频
- `GET /api/v1/style/recommend/{audio_id}`：获取风格推荐
- `POST /api/v1/song/generate`：生成歌曲
- `GET /api/v1/rvc/models`：获取可用 RVC 示例音色
- `GET /api/v1/pipeline/status`：获取流水线状态
- `GET /api/v1/health`：健康检查

## 使用说明

一个典型流程通常是：

1. 上传音频样本
2. 获取推荐风格
3. 输入歌词和风格
4. 指定 RVC 示例音色或用户自定义模型
5. 生成歌曲并获取输出文件 URL

## 说明与注意事项

- 根目录下的 `models/` 已被 `.gitignore` 忽略，不会推送到 GitHub
- 当前项目虽然保留了一些可选方案名称，例如 `musicgen`、`fish_speech`、`diffsinger`、`vits-svs`，但默认主链路仍然是 `ACE-Step + Demucs + RVC`
- 如果要换成你自己的声音，需要把真实训练得到的 RVC `.pth` 模型放入 `models/user_voices/`，并尽量提供对应 `.index`

## 相关文档

- [项目模型文档.md](./项目模型文档.md)
- [进展.md](./进展.md)
- [frontend/README.md](./frontend/README.md)
