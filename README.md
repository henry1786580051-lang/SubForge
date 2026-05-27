# 🔥 SubForge

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

AI 驱动的视频字幕工具 — 语音转录、字幕优化、智能翻译、视频合成，一站完成。

> Web UI + 桌面应用 + CLI，全平台覆盖。

> **👉 [下载桌面应用](https://github.com/henry1786580051-lang/SubForge/releases/latest)**，macOS / Windows 一键安装，开箱即用。

<p align="center">
  <img src="furnace_app_icon_v2.svg" alt="SubForge Logo" width="120" />
</p>

---

## 📑 目录

- [功能特点](#-功能特点)
- [快速开始](#-快速开始)
- [CLI 命令行](#-cli-命令行)
- [ASR 引擎](#-asr-引擎)
- [翻译引擎](#-翻译引擎)
- [LLM 支持](#-llm-支持)
- [Claude Code Skill](#-claude-code-skill)
- [技术栈](#️-技术栈)
- [项目结构](#-项目结构)
- [开发](#-开发)
- [许可证](#-许可证)

---

## 🌟 功能特点

### 🎙️ 语音转录

- **🔊 多引擎 ASR**：支持 Whisper.cpp（本地 GPU）、Whisper API、Faster Whisper、剪映、必剪等多种转录引擎
- **🎯 自动语言检测**：自动识别视频语言，支持中英日韩等 100+ 语种
- **⚡ GPU 加速**：Whisper.cpp 支持 Metal / CUDA 加速，转录速度快数倍
- **📝 长音频分段**：自动切割长音频，分段转录后智能合并

### ✨ 字幕优化

- **🤖 LLM 智能优化**：调用大语言模型优化断句、修正错别字、改善可读性
- **💬 自定义提示词**：可自定义优化指令，控制输出风格
- **🔄 反思纠错**：支持多轮反思机制，提升字幕准确率

### 🌐 智能翻译

- **🧠 LLM 翻译**：使用大模型翻译，上下文理解准确、表达自然
- **🆓 免费翻译**：支持 Bing、Google、DeepLX 等免费翻译引擎
- **📊 双语字幕**：支持原文 + 译文双语对照输出
- **🎯 多语言**：支持中文、英文、日语、韩语等主流语言互译

### 📄 字幕格式

- **📥 多格式导入**：SRT、VTT、ASS 字幕文件导入
- **📤 多格式导出**：SRT、VTT、ASS、TXT、JSON 五种格式
- **📝 在线编辑**：Web 界面直接编辑字幕文本和译文
- **🔀 合并/拆分**：支持字幕段落合并与批量操作

### 🖥️ 双模式运行

- **🌐 Web UI**：Next.js 现代化界面，浏览器直接使用
- **💻 桌面应用**：Electron 封装，原生体验，系统托盘常驻
- **⌨️ CLI 命令行**：`subforge` 命令行工具，支持脚本自动化

---

## 🚀 快速开始

### 方式一：下载桌面应用（推荐）

前往 [Releases](https://github.com/henry1786580051-lang/SubForge/releases/latest) 页面下载安装包：

| 平台 | 文件 |
|------|------|
| **macOS** | `SubForge-*-arm64.dmg` |
| **Windows** | `SubForge-Setup-*.exe` |

> 💡 macOS 首次打开需在「系统设置 → 隐私与安全性」中点击「仍要打开」。

### 方式二：pip 安装（CLI）

```bash
pip install subforge
subforge --help
```

免费功能（必剪语音识别、必应/谷歌翻译）**无需任何配置，安装即用**。

### 方式三：从源码运行（开发者）

<details>
<summary><strong>🍎 macOS / 🐧 Linux</strong></summary>

```bash
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

# 后端
uv sync
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --port 8000 &

# 前端
cd frontend
npm install
npm run dev
```

浏览器打开 **http://localhost:3000** 即可使用。

</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

```powershell
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

# 后端
uv sync
$env:PYTHONPATH="backend"; .venv\Scripts\uvicorn app.main:app --port 8000

# 前端（新终端）
cd frontend
npm install
npm run dev
```

浏览器打开 **http://localhost:3000** 即可使用。

</details>

---

## ⌨️ CLI 命令行

```bash
# 语音转录（免费，无需 API Key）
subforge transcribe video.mp4 --asr bijian

# 字幕翻译（免费必应翻译）
subforge subtitle input.srt --translator bing --target-language en

# 全流程：转录 → 优化 → 翻译 → 合成
subforge process video.mp4 --target-language ja

# 字幕烧录到视频
subforge synthesize video.mp4 -s subtitle.srt

# 下载在线视频
subforge download "https://youtube.com/watch?v=xxx"
```

<details>
<summary>所有 CLI 命令一览</summary>

| 命令 | 说明 |
|------|------|
| `gui` | 打开桌面版。也可以直接运行 `subforge-gui` |
| `transcribe` | 语音转字幕。引擎：`faster-whisper`、`whisper-api`、`bijian`（免费）、`jianying`（免费）、`whisper-cpp` |
| `subtitle` | 字幕优化/翻译。翻译服务：`llm`、`bing`（免费）、`google`（免费） |
| `dub` | 根据字幕生成配音音轨或配音视频 |
| `synthesize` | 字幕烧录到视频（软字幕/硬字幕） |
| `process` | 全流程处理 |
| `download` | 下载 YouTube、B站等平台视频 |
| `config` | 配置管理（`show`、`set`、`get`、`path`、`init`） |

运行 `subforge <命令> --help` 查看完整参数。完整 CLI 文档见 [docs/cli.md](docs/cli.md)。

</details>

---

## 🎙️ ASR 引擎

| 引擎 | 类型 | 说明 |
|------|------|------|
| **Whisper.cpp** | 本地 | 推荐，支持 Metal/CUDA GPU 加速 |
| **Whisper API** | 云端 | OpenAI Whisper API，需 API Key |
| **Faster Whisper** | 本地 | CTranslate2 加速，性能优秀 |
| **剪映 ASR** | 云端 | 剪映同款引擎，中文效果好 |
| **必剪 ASR** | 云端 | B 站必剪引擎，免费使用 |

---

## 🌐 翻译引擎

| 引擎 | 类型 | 说明 |
|------|------|------|
| **LLM 翻译** | 云端 | 推荐，上下文理解准确自然 |
| **Bing** | 免费 | 微软翻译，免费无需配置 |
| **Google** | 免费 | 谷歌翻译 |
| **DeepLX** | 免费 | DeepL 开源替代 |

---

## 🤖 LLM 支持

SubForge 使用 OpenAI 兼容接口，支持任意 LLM 提供商：

| 提供商 | 推荐模型 |
|--------|----------|
| **小米 MiMo** | mimo-v2.5-pro（默认） |
| **OpenAI** | gpt-4o / gpt-4o-mini |
| **DeepSeek** | deepseek-chat |
| **通义千问** | qwen-plus |
| **LM Studio** | 本地模型（免费） |

需要 LLM 功能（字幕优化、大模型翻译）时，配置 API Key：

```bash
subforge config set llm.api_key <your-key>
subforge config set llm.api_base https://api.openai.com/v1
subforge config set llm.model gpt-4o-mini
```

配置优先级：`命令行参数 > 环境变量 > 配置文件 > 默认值`。运行 `subforge config show` 查看当前配置。

---

## 🔧 Claude Code Skill

本项目提供了 [Claude Code Skill](https://code.claude.com/docs/en/skills.md)，让 AI 编程助手可以直接调用 SubForge 处理视频。

安装到 Claude Code：

```bash
mkdir -p ~/.claude/skills/subforge
cp skills/SKILL.md ~/.claude/skills/subforge/SKILL.md
```

然后在 Claude Code 中输入 `/subforge transcribe video.mp4 --asr bijian` 即可使用。

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| [Python 3.10+](https://www.python.org/) | 核心后端语言 |
| [FastAPI](https://fastapi.tiangolo.com/) | Web API 框架 |
| [Next.js 16](https://nextjs.org/) | 前端框架 |
| [React](https://react.dev/) | UI 组件库 |
| [TypeScript](https://www.typescriptlang.org/) | 类型安全 |
| [Zustand](https://zustand-demo.pmnd.rs/) | 状态管理 |
| [Tailwind CSS](https://tailwindcss.com/) | 样式框架 |
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | 本地语音识别 |
| [Electron](https://www.electronjs.org/) | 桌面应用封装 |
| [PyInstaller](https://pyinstaller.org/) | Python 打包 |

---

## 📁 项目结构

```
SubForge/
├── subforge/                  # Python 核心库
│   ├── core/
│   │   ├── asr/               # ASR 引擎（Whisper.cpp / API / Faster Whisper）
│   │   ├── translate/         # 翻译引擎（LLM / Bing / Google / DeepLX）
│   │   ├── llm/               # LLM 调用与日志
│   │   └── subtitle/          # 字幕处理（优化 / 合成）
│   ├── cli/                   # CLI 命令行入口
│   └── config.py              # 全局配置
├── backend/                   # FastAPI 后端
│   └── app/
│       ├── api/               # API 路由
│       └── core/              # 任务管理
├── frontend/                  # Next.js 前端
│   └── src/
│       ├── app/               # 页面
│       ├── components/        # React 组件
│       ├── store/             # Zustand 状态
│       └── lib/               # API 客户端
├── pyproject.toml             # Python 项目配置
└── package.json               # Electron 打包配置
```

---

## 📖 工作原理

```
音视频输入 → 语音识别 → 字幕断句 → LLM 优化 → 翻译 → 视频合成
```

- 词级时间戳 + VAD 语音活动检测，识别准确率高
- LLM 语义理解断句，字幕阅读体验自然流畅
- 上下文感知翻译，支持反思优化机制
- 批量并发处理，效率高

---

## 💻 开发

```bash
git clone https://github.com/henry1786580051-lang/SubForge.git
cd SubForge

# Python 开发
uv sync
uv run subforge --help          # CLI
uv run subforge                 # GUI 桌面版
uv run pyright                  # 类型检查
uv run pytest tests/test_cli/ -q  # 运行测试

# Web 开发
cd frontend
npm install
npm run dev                     # 启动前端 :3000
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的改动 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

---

## 📄 许可证

[GPL-3.0 License](LICENSE)

