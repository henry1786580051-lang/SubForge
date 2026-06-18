# SubForge

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

<p align="center">
  <img src="furnace_app_icon_v2.svg" alt="SubForge Logo" width="120" />
</p>

<p align="center">
  <img src="docs/screenshot.png" alt="SubForge Screenshot" width="800" />
</p>

SubForge 是一个 AI 驱动的视频字幕工具，覆盖转录、断句、优化、翻译、字幕样式与视频合成等流程。它既可以作为桌面/网页工具使用，也可以通过 CLI 和 Python 模块集成到自己的工作流中。

## 能做什么

| 能力 | 说明 |
| --- | --- |
| 语音转文字 | 默认使用 WhisperX 工作流：MLX Whisper 转录 + forced alignment 词级时间轴 + VAD 保守校验 |
| 智能断句 | 使用 LLM 按语义重排字幕，避免机械切分和超长字幕 |
| 字幕优化 | 自动修正错别字、补全标点、去除冗余语气词 |
| 智能翻译 | 支持上下文感知翻译、反思翻译和免费翻译引擎 |
| 双语字幕 | 可导出 SRT、VTT、ASS、TXT、JSON 等格式 |
| 语音合成 | 支持字幕配音与视频合成相关工作流 |
| Web 界面 | 拖拽上传、实时进度、在线编辑、请求日志查看 |

## 核心技术与实测

SubForge 的重点不只是“把语音转成文字”，而是尽量让字幕达到可发布、可阅读、可翻译的状态。当前主流程围绕 Apple Silicon 本地转录、WhisperX 对齐和上下文翻译组织：

### 转录优化

转录前会先对音频做预处理，再使用 MLX Whisper 完成 Apple Silicon 加速转录。文本在对齐前会将数字、单位和符号转为可对齐的口语形式，然后由 WhisperX forced alignment 生成词级时间轴。TEN-VAD 仅对可疑边界做保守校验，避免为了修正少量错误而破坏已经正确的字幕：

```text
原始音频
  -> DeepFilterNet3 可选降噪
  -> MLX Whisper 转录
  -> 数字/单位/符号语音规范化
  -> WhisperX forced alignment
  -> TEN-VAD 时间轴保守校验（Silero VAD 回退）
  -> 词级时间轴
  -> 智能断句与翻译
```

| 技术 | 作用 |
| --- | --- |
| DeepFilterNet3 | 降低车内、户外、咖啡厅等场景的背景噪音，突出人声 |
| MLX Whisper | Apple Silicon 专门优化的本地 Whisper 推理，默认使用本地 MLX 模型 |
| WhisperX forced alignment | 使用独立对齐模型把转录文本落到词级时间轴 |
| 对齐前语音规范化 | 将 `350` 、`mph` 、`kg` 等数字与单位展开为可对齐的口语 token，对齐后恢复原文展示 |
| TEN-VAD | 默认的语音活动检测器，用于校验可疑句首和句尾，不全局覆盖 WhisperX 的正确对齐 |
| Silero VAD | TEN-VAD 不可用或运行失败时的回退方案，保证跨平台可用性 |
| Content Integrity Score | 用语音时长比例监控内容完整性，避免参数过严导致漏转录 |
| Whisper.cpp 兼容通道 | 保留 whisper.cpp 作为备用本地引擎，适合已有 ggml 模型的用户 |

### 可靠性与桌面端

- 转录、断句和翻译任务通过 WebSocket 实时推送进度与中间结果。
- 上传文件、缩略图和导出结果使用独立路径与范围请求处理，避免同名文件或并发任务相互覆盖。
- LLM 请求日志按请求绑定，并发翻译不会错配 prompt 和 response。
- macOS 桌面包内置 FFmpeg、DeepFilterNet3 运行时和 TEN-VAD；Whisper/forced alignment 模型由用户在设置页管理，不重复打包大模型。

### 智能翻译

LLM 翻译会结合上下文处理整段内容，而不是逐句机械直译。对于表达质量要求更高的视频，可以启用反思翻译模式：

```text
初译 -> 反思机翻痕迹和语境问题 -> 重写为更自然的译文
```

示例：

| 阶段 | 内容 |
| --- | --- |
| 初译 | 今天我们驾驶的是全新2026款雷克萨斯ES 350h。 |
| 反思 | “今天我们驾驶的是”偏英文语序，“全新2026款”更像新闻稿。 |
| 重写 | 今天来试试2026新款雷克萨斯ES 350h。 |

### Token 消耗

以下是 21 分钟英文试驾视频的实测数据，模型为小米 MiMo v2.5-pro，流程包含智能分句、纠错优化和智能翻译：

| 阶段 | 调用次数 | Prompt | Completion | 其中 Reasoning | Total | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 智能分句 | 12 | 17,709 | 51,876 | 44,322 | 69,585 | 13.8min |
| 纠错优化 | 21 | 20,660 | 77,424 | 70,844 | 98,084 | 23.2min |
| 智能翻译 | 20 | 13,103 | 16,745 | 10,923 | 29,848 | 5.6min |
| 合计 | 53 | 51,472 | 146,045 | 126,089 | 197,517 | 42.5min |

MiMo v2.5-pro 是推理模型，Reasoning tokens 占 Completion 的主要部分。实际输出约 20K tokens，最终得到 389 段双语字幕。使用 Bing / Google 等免费翻译引擎可以跳过 LLM 翻译阶段；使用 LM Studio、Ollama 等本地模型则可以进一步降低 API 成本。

## 快速开始

### 运行 Web 版本

需要 Python 3.10-3.12、[uv](https://docs.astral.sh/uv/) 和 Node.js 20+。Apple Silicon 用户如需使用默认的 WhisperX + MLX 转录流程，应安装 `whisperx` 和 `denoise` 可选依赖：

```bash
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

uv sync --extra whisperx --extra denoise
PYTHONPATH=backend uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows/Linux 或只使用云端 Whisper API 时，可将安装命令简化为 `uv sync`。

另开一个终端启动前端：

```bash
cd frontend
npm ci
npm run dev
```

打开 <http://localhost:3000> 即可使用。

### 使用 CLI

```bash
uv run subforge --help
uv run subforge doctor
```

常用命令包括：

```bash
# Apple Silicon 本地 MLX Whisper + WhisperX forced alignment
uv run subforge transcribe input.mp4 --asr whisperx --language auto --word-timestamps

# 使用已配置 API 的云端转录
uv run subforge transcribe input.mp4 --asr whisper-api

uv run subforge subtitle input.srt
uv run subforge dub input.srt
```

### 启动桌面版

普通用户应直接从 [GitHub Releases](https://github.com/henry1786580051-lang/SubForge/releases) 下载 DMG 或 Windows 安装包。源码运行当前桌面应用时，需要先生成前端静态文件：

```bash
cd frontend
npm ci
npm run build
cd ..

uv sync --extra whisperx --extra denoise
uv run python launcher.py
```

`subforge-gui` 是旧版 PyQt 界面入口，不代表当前发布的桌面应用。

## 推荐配置

### LLM

智能断句、优化和翻译使用 OpenAI 兼容接口。可以在设置页中配置 API Base、API Key 和模型名称。

| 提供商 | 示例模型 |
| --- | --- |
| 小米 MiMo | `mimo-v2.5-pro` |
| DeepSeek | `deepseek-chat` |
| OpenAI | `gpt-4o` / `gpt-4o-mini` |
| 通义千问 | `qwen-plus` |
| 本地模型 | LM Studio / Ollama 等 OpenAI 兼容服务 |

### ASR

| 引擎 | 适合场景 |
| --- | --- |
| WhisperX + MLX Whisper | 默认推荐；Apple Silicon 本地加速，配合 forced alignment 生成词级时间轴 |
| WhisperX Alignment | forced alignment 模型分类管理，用于英语等语言的词级对齐 |
| Whisper.cpp | 备用本地转录通道，适合已有 ggml 模型的用户 |
| Whisper API | 云端转录，配置简单 |

## 示例

项目内提供了一组 Lexus 试驾视频字幕样例，用来展示 ASR 原始字幕和 SubForge 处理后的差异：

- [ASR 原始输出](examples/lexus_original.srt)
- [断句与翻译后输出](examples/lexus_processed.srt)

处理后的字幕会更短、更自然，并尽量保持完整语义：

```text
官方数据显示，它高速巡航的油耗大概在百公里4.9升左右。
市区大概百公里5.4升，综合下来5.1升左右。
这油耗表现相当惊人。
综合马力有244匹。
```

## 项目结构

```text
SubForge/
├── frontend/        # Next.js Web 界面
├── backend/         # FastAPI 服务
├── subforge/        # Python 核心库、CLI、桌面端
├── docs/            # VitePress 文档
├── resource/        # 字体、图标、翻译、样式资源
├── tests/           # 自动化测试
└── examples/        # 示例字幕
```

## 开发

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

前端：

```bash
cd frontend
npm install
npm run lint
npm run dev
```

文档：

```bash
cd docs
npm install
npm run docs:dev
```

## 文档与链接

- 文档站点：<https://henry1786580051-lang.github.io/SubForge/>
- 问题反馈：<https://github.com/henry1786580051-lang/SubForge/issues>
- 贡献指南：[docs/dev/contributing.md](docs/dev/contributing.md)

## 许可证

本项目基于 [GPL-3.0 License](LICENSE) 发布。
