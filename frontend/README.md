# 前端使用指南

## 启动服务

```bash
# 方式1: 使用main.py
python main.py

# 方式2: 使用uvicorn
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

## 访问界面

打开浏览器访问: http://localhost:8000

## 使用流程

### 步骤1: 上传音频
- 点击上传区域或拖拽文件
- 支持格式: WAV, MP3, M4A
- 建议: 30秒说话 + 1-2分钟清唱

### 步骤2: 选择风格
- AI自动推荐3个最适合的风格
- 显示匹配度和推荐理由
- 可输入自定义风格

### 步骤3: 输入歌词
- 输入您的歌词内容
- 支持多行文本

### 步骤4: 生成歌曲
- 实时显示生成进度
- 包含4个处理阶段

### 步骤5: 下载结果
- 在线试听生成的歌曲
- 下载WAV格式文件
- 可重新生成

## 界面特点

✅ 响应式设计 - 支持手机/平板/电脑
✅ 拖拽上传 - 方便快捷
✅ 实时进度 - 清晰展示
✅ 美观界面 - 渐变色设计
✅ 交互友好 - 流畅动画

## 技术栈

- HTML5
- CSS3 (渐变、动画、响应式)
- JavaScript (原生ES6+)
- FastAPI (后端API)

## 目录结构

```
frontend/
├── templates/
│   └── index.html          # 主页面
├── static/
│   ├── css/
│   │   └── style.css       # 样式
│   └── js/
│       └── app.js          # 交互逻辑
```

## API接口

- `GET /` - 主页
- `POST /api/v1/audio/upload` - 上传音频
- `GET /api/v1/style/recommend/{audio_id}` - 获取推荐
- `POST /api/v1/song/generate` - 生成歌曲
- `GET /api/v1/health` - 健康检查

## 自定义

### 修改颜色主题
编辑 `frontend/static/css/style.css`:
```css
/* 主色调 */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### 添加新功能
编辑 `frontend/static/js/app.js` 添加新的交互逻辑

### 修改布局
编辑 `frontend/templates/index.html` 调整页面结构
