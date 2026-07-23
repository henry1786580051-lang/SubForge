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
| 语音转文字 | WhisperX 在 Apple Silicon 使用 MLX、在 Windows 使用 CTranslate2，并通过 forced alignment 生成词级时间轴 |
| 多人语音 | 可选 pyannote Community-1 说话人分离，支持双人、自动人数和 2–10 人精确约束；固定保留原始音轨以保护较弱说话人的内容 |
| 智能断句 | 使用 LLM 按语义重排字幕，同时校验原文完整性、长度与时间轴 |
| 字幕优化 | 保守修正明显 ASR 错误和标点，不允许 LLM 任意改写正确原文 |
| 智能翻译 | 支持上下文感知、反思翻译、MiniMax Anthropic 接口及 OpenAI 兼容接口 |
| 双语字幕 | 可导出 SRT、VTT、ASS、TXT、JSON 等格式 |
| 语音合成 | 支持字幕配音与视频合成相关工作流 |
| Web 界面 | 拖拽上传、实时进度、在线编辑、请求日志查看 |

## 当前工作流

SubForge 的重点不只是把语音转成文字，而是尽量让时间轴、原文和译文都达到可发布状态。转录层按平台选择 MLX 或 CTranslate2，之后共享 WhisperX 对齐和带完整性校验的上下文翻译流程。

### 转录优化

单人模式可直接使用 DeepFilterNet3 增强音频。多人模式在原始音频上运行 pyannote Community-1，并固定使用原始音轨完成 ASR、时间轴校验和说话人归属，不执行候选降噪或额外的候选 ASR。已知参与人数的访谈建议使用“指定人数”，可减少自动聚类将同一人拆成多个标签的风险。

转录文本会先将数字、单位和符号展开为 forced alignment 更容易识别的口语 token，对齐完成后再恢复原文显示。TEN-VAD 只修正可疑边界，不全局覆盖已经正确的 WhisperX 时间戳：

```text
原始音频
  -> [多人模式] pyannote Community-1 说话人分离
  -> [单人模式] DeepFilterNet3 可选降噪
  -> [多人模式] 固定保留原始音轨
  -> MLX Whisper（Apple Silicon）/ Faster-Whisper（Windows）
  -> 数字/单位/符号语音规范化
  -> WhisperX forced alignment
  -> TEN-VAD 时间轴保守校验（Silero VAD 回退）
  -> 说话人归属与词级时间轴
  -> 智能断句 -> 保守纠错 -> 上下文生成 -> 翻译 -> 最终质量校验
```

| 技术 | 作用 |
| --- | --- |
| [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) | 单人模式可选增强；多人模式自动跳过，以避免压制较弱说话人的语音 |
| [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | Apple Silicon 专门优化的本地 Whisper 推理，默认使用本地 MLX 模型 |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | Windows 上为 WhisperX 提供 CTranslate2/CUDA 或 CPU 转录路径 |
| [WhisperX forced alignment](https://github.com/m-bain/whisperX) | 按源语言自动匹配独立对齐模型，把转录文本落到词级时间轴；设置页可搜索、下载并检查各语言模型 |
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
- macOS 桌面包内置 FFmpeg、DeepFilterNet3 运行时和 TEN-VAD；Whisper、forced alignment 与说话人模型由用户按需管理，不重复打包大模型。词级对齐模型默认按源语言自动选择，无需手动记忆模型 ID。

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

## 五人访谈实测案例

以下数据来自 Matt Damon、Anne Hathaway、Tom Holland、Christopher Nolan 与主持人的完整访谈。测试日期为 2026-07-22，流程覆盖 MLX 转录、WhisperX forced alignment、pyannote Community-1 五人约束、智能断句、保守纠错、MiniMax M3 翻译和反思校验。片中包含电影原片段，因此这里的“五人”是对访谈参与者数量的约束，并不等同于声纹身份识别。该案例的候选比较最终选择了原音频；当前版本因此在多人模式中直接保留原音频。

| 配置 | 实测值 |
| --- | --- |
| 原始视频 | [“How!” Matt Damon, Anne Hathaway & Tom Holland on Christopher Nolan’s Film Method](https://www.youtube.com/watch?v=9rO0FGivAvQ) |
| 视频时长 | 18:03 |
| ASR / 对齐 | MLX Large V3 FP16 / WhisperX forced alignment |
| 说话人分离 | pyannote Community-1，指定 5 人 |
| 多人音频策略 | 固定保留原音频，不执行候选降噪与额外候选 ASR |
| 转录结果 | 3,777 条词级片段 / 3,785 个词 |
| 翻译模型 | `MiniMax-M3` |
| 协议 / Base URL | Anthropic / `https://api.minimaxi.com/anthropic` |
| 批量 / 并发 | 10 / 10 |
| 目标语言 | 简体中文 |
| 最终处理日志跨度 | 约 14:43，包含限流等待与定向复核 |
| 处理结果 | 443 条中英双语字幕 |

### 多人分离结果

Community-1 在指定五人模式下分配了 3,767 个词，另有 18 个词因置信度不足保持未分配。`Speaker 1–5` 只表示稳定的声纹聚类，不推断或展示人物姓名；最终双语字幕也不会写入说话人标签。

| 聚类 | 词数 |
| --- | ---: |
| Speaker 1 | 1,044 |
| Speaker 2 | 1,004 |
| Speaker 3 | 542 |
| Speaker 4 | 490 |
| Speaker 5 | 687 |
| 未可靠分配 | 18 |

### Token 与缓存统计

下表统计最终采用的完整处理轮次及随后针对可疑错位执行的定向校验，不包含此前被放弃的实验轮次。MiniMax M3 遇到 429 后保持任务存活，因而“请求尝试”包含限流后的重试。

| 阶段 | 请求尝试 | 成功响应 | 限流/重试 | 输入 Token | 缓存读取 | 读取占比 | 输出 Token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 智能断句 | 149 | 130 | 19 | 49,356 | 37,779 | 76.54% | 10,091 |
| 保守纠错 | 77 | 47 | 30 | 43,455 | 15,877 | 36.54% | 6,705 |
| 翻译、反思与定向复核 | 182 | 147 | 35 | 209,140 | 141,158 | 67.49% | 74,489 |
| **合计** | **408** | **324** | **84** | **301,951** | **194,814** | **64.52%** | **91,285** |

输入与输出合计记录 **393,236 Token**。缓存读取占比采用服务端返回的 `cache_read_input_tokens / input_tokens` 计算；`cache_creation_input_tokens` 为 0，因此该比例不代表显式 Prompt Cache 的固定命中率，也不应直接作为计费折扣估算。

### 结果校验

| 指标 | 实测值 |
| --- | ---: |
| 最终字幕数 | 443 |
| 原始转录与最终英文的规范化序列相似度 | 98.58% |
| 中文译文字符数 | 5,549 |
| 空译文 / 编辑占位语 / 思考泄漏 | 0 / 0 / 0 |
| 时间轴重叠 / 非法区间 | 0 / 0 |
| 相邻译文包含式重复 | 0 |
| 中文译文逗号/句号残留 | 0 |

原文相似度用于监测断句和保守纠错是否吞词，不代表说话人识别率或 ASR 语义准确率。本次成品保留了转录主体内容；MiniMax M3 对选角语境中的 `pass` 有一处持续歧义，最终样例经过人工语义复核并修正，没有为该个例加入可能误伤其他视频的硬编码规则。

完整文件可直接检查：

- [输入：五人分离词级字幕](examples/five_speaker_interview_word_timestamps.srt)
- [输出：MiniMax M3 五人访谈双语字幕](examples/five_speaker_interview_minimax_m3_processed.srt)
- [报告：说话人词量与时间范围](examples/five_speaker_interview_report.json)

## 快速开始

### 运行 Web 版本

需要 Python 3.10-3.12、[uv](https://docs.astral.sh/uv/) 和 Node.js 20+。Apple Silicon 用户如需使用默认的 WhisperX + MLX 转录流程，应安装 `whisperx` 和 `denoise` 可选依赖：

```bash
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

uv sync --extra whisperx --extra denoise
PYTHONPATH=backend uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows 使用 WhisperX、forced alignment 或 Community-1 时同样需要 `--extra whisperx`；只使用 Whisper.cpp 或云端 Whisper API 时可简化为 `uv sync`。

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
| WhisperX | Apple Silicon 使用 MLX，Windows 使用 CTranslate2/CUDA 或 CPU；均配合 forced alignment 生成词级时间轴 |
| WhisperX Alignment | 按源语言管理 41 种 forced alignment 模型，支持自动匹配、下载状态、搜索和手动覆盖 |
| Whisper.cpp | 备用本地转录通道，适合已有 ggml 模型的用户 |
| Whisper API | 云端转录，配置简单 |
| pyannote Community-1 | 多人模式；需先获得模型访问权限并在设置页下载到本地模型目录 |

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
