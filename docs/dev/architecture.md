# 架构设计

SubForge 的系统架构设计。

## 技术栈

- **桌面 UI**: Next.js/React 静态前端 + FastAPI 本地后端 + pywebview 宿主
- **ASR 引擎**: Apple Silicon 默认使用 MLX Whisper + WhisperX forced alignment；Whisper.cpp、FasterWhisper 和 Whisper API 为兼容通道
- **音频边界**: DeepFilterNet3 可选降噪，TEN-VAD/Silero VAD 做保守边界校验
- **LLM 集成**: OpenAI/DeepSeek/Gemini/Ollama 等
- **视频处理**: FFmpeg

## 核心模块

### 1. 领域核心 (`subforge/core/`)

包含 ASR、断句、优化、翻译和字幕数据结构。该层不得依赖 FastAPI、React、PyQt 或全局任务管理器。

ASR 的隔离进程通过 `subforge/core/asr/worker_runtime.py` 共享原子消息写入、日志读取和退出升级策略；模型推理与时间轴算法仍由各引擎实现。

### 2. 应用层 (`subforge/application/`)

编排长任务所需的进度、实时预览、取消和用户介入接口。`PipelineContext` 只描述能力，不知道任务来自网页、命令行还是其他宿主。

### 3. 配置层 (`subforge/settings/`)

`AppSettings` 是运行设置的规范模型；FastAPI 与 CLI 通过适配器读取旧格式，避免各入口继续维护不同默认值。

### 4. 接口适配层 (`frontend/`, `backend/`, `subforge/cli/`)

FastAPI 路由只负责输入输出和调用应用层，`backend/app/services/` 将全局任务管理器适配为 `PipelineContext`。React 的静态目录和质量统计位于 `frontend/src/features/`，页面组件负责交互组合。

### 5. 桌面宿主 (`launcher.py`)

Next.js 负责导入、转录、翻译与字幕编辑界面；FastAPI 负责本地任务、文件和实时进度接口；pywebview 将两者封装为桌面应用。

`subforge/ui/` 是冻结的旧版 PyQt 兼容界面，不是当前发布包的主界面。新功能不得依赖该目录，桌面启动器也不得导入它。

## 依赖方向

```
React / CLI / FastAPI
          ↓
接口适配与应用服务
          ↓
领域核心 + 规范配置
          ↓
模型、FFmpeg、翻译服务等外部依赖
```

自动架构测试会阻止 `subforge/core/`、`subforge/application/` 和 `subforge/settings/` 反向依赖接口层。

## 数据流

```
视频/音频 → 音频预处理 → MLX Whisper → forced alignment → VAD 边界校验 → ASRData → 分割 → 优化 → 翻译 → 双语字幕
```

主流程说明和安装方式请参考仓库根目录的 `README.md`。

---

相关文档：
- [API 文档](/dev/api)
- [贡献指南](/dev/contributing)
