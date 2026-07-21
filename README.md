# SubForge

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-009688?logo=fastapi&logoColor=white)
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
| 语音转文字 | 默认使用 MLX Whisper + WhisperX forced alignment，生成词级时间轴并由 VAD 保守校验 |
| 多人语音 | 可选 pyannote Community-1 说话人分离，并用自适应降噪保护较弱说话人的内容 |
| 智能断句 | 使用 LLM 按语义重排字幕，同时校验原文完整性、长度与时间轴 |
| 字幕优化 | 保守修正明显 ASR 错误和标点，不允许 LLM 任意改写正确原文 |
| 智能翻译 | 支持上下文感知、反思翻译、MiniMax Anthropic 接口及 OpenAI 兼容接口 |
| 双语字幕 | 可导出 SRT、VTT、ASS、TXT、JSON 等格式 |
| 语音合成 | 支持字幕配音与视频合成相关工作流 |
| Web 界面 | 拖拽上传、实时进度、在线编辑、请求日志查看 |

## 当前工作流

SubForge 的重点不只是把语音转成文字，而是尽量让时间轴、原文和译文都达到可发布状态。默认流程围绕 Apple Silicon 本地转录、WhisperX 对齐和带完整性校验的上下文翻译组织。

### 转录优化

单人模式可直接使用 DeepFilterNet3 增强音频。多人模式先在原始音频上运行 pyannote Community-1，再抽取多名说话人的代表片段，对原音频和多档轻度降噪结果进行校准；只有在没有损失说话人覆盖时才采用降噪版本。

转录文本会先将数字、单位和符号展开为 forced alignment 更容易识别的口语 token，对齐完成后再恢复原文显示。TEN-VAD 只修正可疑边界，不全局覆盖已经正确的 WhisperX 时间戳：

```text
原始音频
  -> [多人模式] pyannote Community-1 说话人分离
  -> DeepFilterNet3 可选/自适应降噪
  -> MLX Whisper 转录（Apple Silicon）
  -> 数字/单位/符号语音规范化
  -> WhisperX forced alignment
  -> TEN-VAD 时间轴保守校验（Silero VAD 回退）
  -> 说话人归属与词级时间轴
  -> 智能断句 -> 保守纠错 -> 上下文生成 -> 翻译 -> 最终质量校验
```

| 技术 | 作用 |
| --- | --- |
| [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) | 单人模式直接增强；多人模式按说话人覆盖率校准降噪强度，必要时保留原音频 |
| [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | Apple Silicon 专门优化的本地 Whisper 推理，默认使用本地 MLX 模型 |
| [WhisperX forced alignment](https://github.com/m-bain/whisperX) | 使用独立对齐模型把转录文本落到词级时间轴 |
| 对齐前语音规范化 | 将 `350`、`mph`、`kg` 等数字与单位展开为口语 token，对齐后恢复原文展示 |
| [TEN-VAD](https://github.com/TEN-framework/ten-vad) | 默认的语音活动检测器，用于校验可疑句首和句尾，不全局覆盖 WhisperX 的正确对齐 |
| Silero VAD | TEN-VAD 不可用或运行失败时的回退方案，保证跨平台可用性 |
| [pyannote Community-1](https://github.com/pyannote/pyannote-audio) | 可选的本地说话人分离；支持固定双人或自动人数，不把说话人标签写入最终字幕 |
| [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) 兼容通道 | 备用本地引擎，适合已有 GGML 模型的用户 |

### 断句与翻译

LLM 处理不是单次自由生成。每个阶段都有结构化输出、键完整性和内容约束，失败时只重试受影响的批次或单条字幕：

```text
语义断句（原文字符完整性校验）
  -> 保守纠错（数字、专有名词和原文覆盖校验）
  -> 全局上下文摘要
  -> 批量翻译（默认批量/并发均可设为 10）
  -> 可选反思重写
  -> 漏译、占位语、思考内容、跨条数字错配检查
  -> 仅对失败条目回退重译
  -> 中文译文逗号/句号无 LLM 清理
```

- MiniMax 使用官方 Anthropic 兼容协议，原生区分 `thinking` 与最终 `text`，防止思考内容进入字幕。
- 其他模型继续使用 OpenAI 兼容协议；Base URL 和 API Key 按服务商分别保存。
- 翻译结果不得出现“合并至上一条”“同上”“省略”等编辑占位语，也不能缺少批次中的任何索引。
- 数字与专有名词校验以防止跨字幕错配为主，不要求中英文逐词对应，避免损害自然翻译质量。
- MiniMax M3 遇到 HTTP 429 时保持任务存活并等待恢复；其他服务使用有上限的指数退避重试。

### 可靠性与桌面端

- 转录、断句和翻译任务通过 WebSocket 实时推送进度与中间结果。
- 上传文件、缩略图和导出结果使用独立路径与范围请求处理，避免同名文件或并发任务相互覆盖。
- LLM 日志按任务和阶段聚合，仍可展开查看单次请求；并发翻译不会错配 prompt 和 response。
- 翻译阶段增量保存中间结果，最终校验失败时不会丢失已经完成的字幕。
- macOS 桌面包内置 FFmpeg、DeepFilterNet3 运行时和 TEN-VAD；Whisper、forced alignment 与说话人模型由用户管理，不重复打包大模型。

## Audi Q3 实测案例

以下数据来自 2026 Audi Q3 英文试驾视频的完整处理：智能断句、保守纠错、上下文生成、翻译和反思模式。测试日期为 2026-07-19，API 的实时负载、限流与计费规则会变化，数据用于说明量级，不代表固定成本或速度。

| 配置 | 实测值 |
| --- | --- |
| 原始视频 | [2026 Audi Q3 - New Turbo Compact SUV Real World City Commute](https://www.youtube.com/watch?v=dY6D-wNBEFM) |
| 视频时长 | 34:47 |
| 输入 | 5,803 条英语词级 SRT 片段 |
| 模型 | `MiniMax-M3` |
| 协议 / Base URL | Anthropic / `https://api.minimaxi.com/anthropic` |
| 批量 / 并发 | 10 / 10 |
| 目标语言 | 简体中文 |
| LLM 请求日志跨度 | 约 5:16 |
| 处理结果 | 629 条中英双语字幕 |

### Token 与缓存统计

| 阶段 | 请求尝试 | 成功响应 | 限流/重试 | 输入 Token | 缓存读取 | 读取占比 | 输出 Token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 智能断句 | 38 | 38 | 0 | 44,120 | 18,546 | 42.04% | 12,168 |
| 保守纠错 | 102 | 64 | 38 | 73,334 | 15,402 | 21.00% | 9,921 |
| 上下文生成 | 1 | 1 | 0 | 3,014 | 128 | 4.25% | 1,372 |
| 翻译与反思 | 117 | 114 | 3 | 395,311 | 141,647 | 35.83% | 98,249 |
| **合计** | **258** | **217** | **41** | **515,779** | **175,723** | **34.07%** | **121,710** |

输入与输出合计记录 **637,489 Token**。“限流/重试”是 HTTP 请求级尝试，不是 41 条字幕失败。MiniMax M3 的等待策略在 429 后继续任务，最终文件没有空译文。本次 API 返回的 `cache_creation_input_tokens` 为 **0**；表中的 34.07% 是服务端报告的 `cache_read_input_tokens / input_tokens`，主要来自自动复用和重试，不应理解为 M3 主动 Prompt Cache 的稳定命中率。当前显式 Prompt Cache 只对 MiniMax M2 系列启用，实际计费应以服务商账单口径为准。

### 结果校验

| 指标 | 当前 Anthropic 接入 |
| --- | ---: |
| 最终字幕数 | 629 |
| 原文规范化序列相似度 | 99.72% |
| 中文译文字符数 | 9,319 |
| 缓存读取占比 | 34.07% |
| 空译文 / 编辑占位语 / 思考泄漏 | 0 / 0 / 0 |
| 时间轴重叠 / 非法区间 | 0 / 0 |
| 超过 18 个英文词的字幕 | 0 |
| 中文译文逗号/句号残留 | 0 |

原文相似度由输入与输出英文合并后转为小写、移除非字母数字字符，再使用序列匹配计算。该指标用于发现断句/纠错阶段的吞词或增词，不代表 ASR 文本本身的语义准确率。

完整文件可直接检查：

- [输入：词级 ASR 字幕](examples/audi_q3_word_timestamps.srt)
- [输出：MiniMax M3 Anthropic 双语字幕](examples/audi_q3_minimax_m3_anthropic_processed.srt)

最终字幕保持中文在上、英文在下，并只移除中文译文中的逗号和句号；英文标点及时间轴不受影响。

## 快速开始

### 运行 Web 版本

需要 Python 3.10-3.12、[uv](https://docs.astral.sh/uv/) 和 Node.js 20+。Apple Silicon 用户如需使用默认的 WhisperX + MLX 转录流程，应安装 `whisperx` 和 `denoise` 可选依赖：

```bash
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

uv sync --extra whisperx --extra denoise
PYTHONPATH=backend uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows/Linux 或只使用云端 Whisper API 时，可将安装命令简化为 `uv sync`。如需本地 FasterWhisper，使用 `uv sync --extra faster-whisper`；如需多人语音识别，再加 `--extra diarization`。说话人分离可与 Whisper.cpp、FasterWhisper、Whisper API 或 Apple Silicon 上的 WhisperX/MLX 转录组合使用。Whisper.cpp 的单个 `ggml-*.bin` 与 FasterWhisper 的 CTranslate2 模型目录格式不同，不能混用。

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

智能断句、优化和翻译支持 MiniMax Anthropic 协议及 OpenAI 兼容协议。设置页会按服务商保存 Base URL、API Key 和模型名称，切换服务商不会复用另一家的密钥。

| 提供商 | 协议 | 示例模型 / 说明 |
| --- | --- | --- |
| MiniMax | Anthropic | `MiniMax-M3`；原生思考分离，429 持续等待；M2 系列支持显式 Prompt Cache |
| 小米 MiMo | OpenAI 兼容 | `mimo-v2.5-pro` |
| DeepSeek | OpenAI 兼容 | `deepseek-chat` |
| OpenAI | OpenAI | 选择账户当前可用模型 |
| 通义千问 | OpenAI 兼容 | `qwen-plus` |
| 本地模型 | OpenAI 兼容 | LM Studio / Ollama 等服务 |

MiniMax 推荐 Base URL 为 `https://api.minimaxi.com/anthropic`。SubForge 会自动使用 `/v1/messages`，不要在 Base URL 中重复添加该路径。

### ASR

| 引擎 | 适合场景 |
| --- | --- |
| WhisperX + MLX Whisper | 默认推荐；Apple Silicon 本地加速，配合 forced alignment 生成词级时间轴 |
| WhisperX Alignment | 独立管理 forced alignment 模型，用于英语等语言的词级对齐 |
| Whisper.cpp | 备用本地转录通道，适合已有 ggml 模型的用户 |
| Whisper API | 云端转录，配置简单 |
| pyannote Community-1 | 跨平台多人模式，可搭配任一 ASR 引擎；需先获得模型访问权限并在设置页下载到本地模型目录 |

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
npm ci
npm run lint
npm run dev
```

文档：

```bash
cd docs
npm ci
npm run docs:dev
```

## 文档与链接

- 文档站点：<https://henry1786580051-lang.github.io/SubForge/>
- 问题反馈：<https://github.com/henry1786580051-lang/SubForge/issues>
- 贡献指南：[docs/dev/contributing.md](docs/dev/contributing.md)

## 许可证

本项目基于 [GPL-3.0 License](LICENSE) 发布。
