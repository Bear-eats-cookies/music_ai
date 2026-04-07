// 全局变量
let audioFile = null;
let audioId = null;
let selectedStyle = null;
let recommendations = [];
let availableRvcModels = [];
let selectedRvcModelName = null;

// API基础URL
const API_BASE = '/api/v1';

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initRvcModels();
    initGenerate();
});

async function initRvcModels() {
    const select = document.getElementById('rvcModelSelect');
    const help = document.getElementById('rvcModelHelp');

    select.innerHTML = '<option value="">加载中...</option>';
    select.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/rvc/models`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || '获取示例音色失败');
        }

        availableRvcModels = data.models || [];
        renderRvcModelOptions(availableRvcModels, data.default_model_name);
        selectedRvcModelName = select.value || null;

        if (availableRvcModels.length > 0) {
            const defaultName = data.default_model_name || availableRvcModels[0].name;
            help.textContent = `默认使用官方示例音色：${defaultName}`;
        } else {
            help.textContent = '当前未发现官方示例音色，可继续生成但不会强制指定示例音色。';
        }
    } catch (error) {
        availableRvcModels = [];
        select.innerHTML = '<option value="">未加载到示例音色</option>';
        select.disabled = true;
        help.textContent = `示例音色列表加载失败：${error.message}`;
    }

    select.addEventListener('change', () => {
        selectedRvcModelName = select.value || null;
    });
}

function renderRvcModelOptions(models, defaultModelName) {
    const select = document.getElementById('rvcModelSelect');

    if (!models || models.length === 0) {
        select.innerHTML = '<option value="">未发现示例音色</option>';
        select.disabled = true;
        return;
    }

    select.innerHTML = models.map((model) => {
        const suffix = model.is_default ? '（默认）' : '';
        const selected = model.name === defaultModelName ? ' selected' : '';
        return `<option value="${model.name}"${selected}>${model.name}${suffix}</option>`;
    }).join('');
    select.disabled = false;
}

// ========== 步骤1: 上传音频 ==========
function initUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('audioFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');

    // 点击上传区域
    uploadArea.addEventListener('click', () => fileInput.click());

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        audioFile = e.target.files[0];
        if (audioFile) {
            uploadArea.innerHTML = `
                <div class="upload-icon">✓</div>
                <p>已选择: ${audioFile.name}</p>
                <p style="color: #666; font-size: 0.9em;">${(audioFile.size / 1024 / 1024).toFixed(2)} MB</p>
            `;
            uploadBtn.classList.remove('hidden');
        }
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        audioFile = e.dataTransfer.files[0];
        if (audioFile) {
            fileInput.files = e.dataTransfer.files;
            uploadArea.innerHTML = `
                <div class="upload-icon">✓</div>
                <p>已选择: ${audioFile.name}</p>
            `;
            uploadBtn.classList.remove('hidden');
        }
    });

    // 上传按钮
    uploadBtn.addEventListener('click', async () => {
        if (!audioFile) return;

        uploadBtn.disabled = true;
        uploadBtn.textContent = '上传中...';
        uploadStatus.className = 'status';
        uploadStatus.textContent = '正在上传音频...';
        uploadStatus.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('file', audioFile);
            formData.append('audio_type', 'mixed');

            const response = await fetch(`${API_BASE}/audio/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                audioId = data.audio_id;
                uploadStatus.className = 'status success';
                uploadStatus.textContent = '✓ 上传成功！正在分析...';
                
                // 获取风格推荐
                await getRecommendations();
            } else {
                throw new Error(data.detail || '上传失败');
            }
        } catch (error) {
            uploadStatus.className = 'status error';
            uploadStatus.textContent = `✗ ${error.message}`;
            uploadBtn.disabled = false;
            uploadBtn.textContent = '重新上传';
        }
    });
}

// ========== 步骤2: 获取风格推荐 ==========
async function getRecommendations() {
    try {
        const response = await fetch(`${API_BASE}/style/recommend/${audioId}`);
        const data = await response.json();

        if (response.ok) {
            recommendations = data.recommendations;
            displayRecommendations(recommendations);
            showStep(2);
        } else {
            throw new Error('获取推荐失败');
        }
    } catch (error) {
        alert(`错误: ${error.message}`);
    }
}

function displayRecommendations(recs) {
    const container = document.getElementById('recommendations');
    container.innerHTML = recs.map((rec, index) => `
        <div class="recommendation-card" data-style="${rec.style}" onclick="selectStyle('${rec.style}', ${index})">
            <h3>${getStyleName(rec.style)}</h3>
            <p class="confidence">匹配度: ${(rec.confidence * 100).toFixed(0)}%</p>
            <p style="font-size: 0.9em; color: #666; margin-top: 10px;">${rec.reason}</p>
        </div>
    `).join('');
}

function selectStyle(style, index) {
    selectedStyle = style;
    
    // 更新选中状态
    document.querySelectorAll('.recommendation-card').forEach((card, i) => {
        card.classList.toggle('selected', i === index);
    });
    
    // 显示下一步
    setTimeout(() => showStep(3), 300);
}

function getStyleName(style) {
    const names = {
        'pop_ballad': '流行抒情',
        'folk_acoustic': '民谣',
        'r&b_soul': 'R&B/Soul',
        'rock': '摇滚'
    };
    return names[style] || style;
}

// ========== 步骤3: 生成歌曲 ==========
function initGenerate() {
    const generateBtn = document.getElementById('generateBtn');
    const lyricsInput = document.getElementById('lyrics');
    const customStyleInput = document.getElementById('customStyle');

    generateBtn.addEventListener('click', async () => {
        const lyrics = lyricsInput.value.trim();
        const customStyle = customStyleInput.value.trim();

        if (!lyrics) {
            alert('请输入歌词');
            return;
        }

        // 使用自定义风格或推荐风格
        const finalStyle = customStyle || selectedStyle;

        if (!finalStyle) {
            alert('请选择或输入音乐风格');
            return;
        }

        generateBtn.disabled = true;
        generateBtn.textContent = '生成中...';

        showStep(4);
        await generateSong(lyrics, finalStyle);
    });
}

async function generateSong(lyrics, style) {
    try {
        // 模拟进度
        simulateProgress();

        const response = await fetch(`${API_BASE}/song/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                audio_id: audioId,
                style: style,
                lyrics: lyrics,
                rvc_model_name: selectedRvcModelName
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data.result);
        } else {
            throw new Error(data.detail || '生成失败');
        }
    } catch (error) {
        alert(`错误: ${error.message}`);
        document.getElementById('generateBtn').disabled = false;
        document.getElementById('generateBtn').textContent = '🎵 生成歌曲';
    }
}

function simulateProgress() {
    const steps = [
        { id: 'progress1', text: '✓ 音频预处理', percent: 20 },
        { id: 'progress2', text: '✓ 声音克隆', percent: 40 },
        { id: 'progress3', text: '✓ 音乐生成', percent: 70 },
        { id: 'progress4', text: '✓ 后处理混音', percent: 100 }
    ];

    let currentStep = 0;

    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            const step = steps[currentStep];
            
            // 更新进度条
            document.getElementById('progressFill').style.width = step.percent + '%';
            document.getElementById('progressText').textContent = step.percent + '%';
            
            // 更新步骤状态
            const stepEl = document.getElementById(step.id);
            stepEl.textContent = step.text;
            stepEl.classList.add('completed');
            
            currentStep++;
        } else {
            clearInterval(interval);
        }
    }, 2000);
}

// ========== 步骤5: 显示结果 ==========
function displayResult(result) {
    showStep(5);

    const audio = document.getElementById('resultAudio');
    const finalSongUrl = result.final_song_url || result.final_song_path;
    audio.src = finalSongUrl;

    document.getElementById('resultStyle').textContent = getStyleName(result.selected_style);
    document.getElementById('resultVoiceModel').textContent = getVoiceModelName(result);
    const durationSeconds = result.metadata?.generation_params?.duration;
    document.getElementById('resultDuration').textContent = durationSeconds
        ? `${Math.floor(durationSeconds / 60)}:${String(durationSeconds % 60).padStart(2, '0')}`
        : '未知';

    // 下载按钮
    document.getElementById('downloadBtn').onclick = () => {
        const a = document.createElement('a');
        a.href = finalSongUrl;
        a.download = `my_song_${Date.now()}.wav`;
        a.click();
    };

    // 重新生成按钮
    document.getElementById('restartBtn').onclick = () => {
        location.reload();
    };
}

function getVoiceModelName(result) {
    const requestedName = result.metadata?.requested_voice_model_name;
    if (requestedName) {
        return requestedName;
    }

    const modelPath = result.metadata?.voice_profile?.voice_model_path;
    if (!modelPath) {
        return '未指定';
    }

    const fileName = modelPath.split(/[\\/]/).pop() || modelPath;
    return fileName.replace(/\.[^.]+$/, '');
}

// ========== 工具函数 ==========
function showStep(stepNumber) {
    // 隐藏所有步骤
    document.querySelectorAll('.step').forEach(step => {
        step.classList.add('hidden');
    });

    // 显示指定步骤
    document.getElementById(`step${stepNumber}`).classList.remove('hidden');

    // 滚动到顶部
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
