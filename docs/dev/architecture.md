# 架构设计

SubForge 的系统架构设计。

## 技术栈

- **桌面 UI**: Next.js/React 静态前端 + FastAPI 本地后端 + pywebview 宿主
- **ASR 引擎**: Apple Silicon 默认使用 MLX Whisper + WhisperX forced alignment；Whisper.cpp、FasterWhisper 和 Whisper API 为兼容通道
- **音频边界**: DeepFilterNet3 可选降噪，TEN-VAD/Silero VAD 做保守边界校验
- **LLM 集成**: OpenAI/DeepSeek/Gemini/Ollama 等
- **视频处理**: FFmpeg

## 核心模块

### 1. ASR 模块 (`subforge/core/asr/`)

语音识别模块，支持多种 ASR 引擎。

### 2. 字幕处理模块 (`subforge/core/split/`, `subforge/core/optimize/`)

字幕分割和优化模块，使用 LLM 进行智能处理。

### 3. 翻译模块 (`subforge/core/translate/`)

字幕翻译模块，支持多种翻译服务。

### 4. 桌面应用 (`frontend/`, `backend/`, `launcher.py`)

Next.js 负责导入、转录、翻译与字幕编辑界面；FastAPI 负责本地任务、文件和实时进度接口；pywebview 将两者封装为桌面应用。

`subforge/ui/` 是兼容保留的旧版 PyQt 界面，不是当前发布包的主界面。

## 数据流

```
视频/音频 → 音频预处理 → MLX Whisper → forced alignment → VAD 边界校验 → ASRData → 分割 → 优化 → 翻译 → 双语字幕
```

主流程说明和安装方式请参考仓库根目录的 `README.md`。

---

相关文档：
- [API 文档](/dev/api)
- [贡献指南](/dev/contributing)
