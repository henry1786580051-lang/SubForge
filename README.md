# SubForge

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

<p align="center">
  <img src="furnace_app_icon_v2.svg" alt="SubForge Logo" width="120" />
</p>

SubForge 是一个 AI 驱动的视频字幕工具，覆盖转录、断句、优化、翻译、字幕样式与视频合成等流程。它既可以作为桌面/网页工具使用，也可以通过 CLI 和 Python 模块集成到自己的工作流中。

## 能做什么

| 能力 | 说明 |
| --- | --- |
| 语音转文字 | 支持 Whisper.cpp、Whisper API、Faster Whisper 等 ASR 引擎 |
| 智能断句 | 使用 LLM 按语义重排字幕，避免机械切分和超长字幕 |
| 字幕优化 | 自动修正错别字、补全标点、去除冗余语气词 |
| 智能翻译 | 支持上下文感知翻译、反思翻译和免费翻译引擎 |
| 双语字幕 | 可导出 SRT、VTT、ASS、TXT、JSON 等格式 |
| 语音合成 | 支持字幕配音与视频合成相关工作流 |
| Web 界面 | 拖拽上传、实时进度、在线编辑、请求日志查看 |

## 快速开始

### 运行 Web 版本

```bash
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

uv sync
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --port 8000
```

另开一个终端启动前端：

```bash
cd frontend
npm install
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
uv run subforge transcribe input.mp4
uv run subforge subtitle input.srt
uv run subforge dub input.srt
```

### 启动桌面版

```bash
uv run subforge-gui
```

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
| Whisper.cpp | 本地转录，支持 Metal / CUDA 加速 |
| Whisper API | 云端转录，配置简单 |
| Faster Whisper | 本地高速转录，适合批量任务 |

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
