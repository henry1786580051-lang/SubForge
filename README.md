# 🔥 SubForge

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

AI 驱动的视频字幕工具 — 语音转录、字幕优化、智能翻译、视频合成，一站完成。

> **👉 [下载桌面应用](https://github.com/henry1786580051-lang/SubForge/releases/latest)**，macOS / Windows 一键安装，开箱即用。

<p align="center">
  <img src="furnace_app_icon_v2.svg" alt="SubForge Logo" width="120" />
</p>

---

## 📑 目录

- [核心特性](#-核心特性)
- [智能断句](#-智能断句)
- [智能翻译](#-智能翻译)
- [Web 界面](#-web-界面)
- [快速开始](#-快速开始)
- [LLM 配置](#-llm-配置)
- [ASR 引擎](#-asr-引擎)
- [技术栈](#️-技术栈)
- [项目结构](#-项目结构)
- [许可证](#-许可证)

---

## 🌟 核心特性

| 特性 | 说明 |
|------|------|
| **智能断句** | LLM 按语义自然分段，而非机械切割，字幕阅读体验流畅 |
| **智能翻译** | 上下文感知翻译，理解整段语境，输出地道表达而非逐词直译 |
| **纠错优化** | 自动去除语气词、修正错别字、规范标点格式 |
| **双语字幕** | 原文 + 译文对照输出，支持 SRT / VTT / ASS / TXT / JSON |
| **多引擎 ASR** | Whisper.cpp（GPU 加速）、Whisper API、Faster Whisper、剪映、必剪 |
| **Web 界面** | 浏览器直接使用，拖拽上传、实时进度、在线编辑字幕 |
| **桌面应用** | Electron 封装，原生体验，支持 macOS 和 Windows |

---

## 🧠 智能断句

语音识别输出的原始文本通常是连续的长句，直接作为字幕会非常难读。SubForge 使用 LLM 按语义自然断点重新分段：

**ASR 原始输出**：
```
大家好今天我们带来的3d创意设计作品是进制演示器我是来自中山大学附属中学的方若涵
我是陈欣然我们这一次作品介绍分为三个部分第一个部分提出问题第二个部分解决方案
```

**LLM 智能断句后**：
```
大家好
今天我们带来的3d创意设计作品是
进制演示器
我是来自中山大学附属中学的方若涵
我是陈欣然
我们这一次作品介绍分为三个部分
第一个部分提出问题
第二个部分解决方案
```

断句规则：
- 在语义自然断点处分段，不截断完整语义
- 中文每段 ≤ 20 字，英文每段 ≤ 10 词
- 保持原文不变，仅插入分段

---

## 🌐 智能翻译

与逐句机翻不同，SubForge 的 LLM 翻译会结合上下文理解整段内容：

| 原文（俄语） | LLM 翻译 |
|------|----------|
| Сегодня ещё раз обращаюсь ко всем гражданам России | 今天我再次向全体俄罗斯公民呼吁 |
| Благодарю вас за выдержку, сплочённость и эту гражданскую солидарность | 感谢大家的坚韧、团结以及这种公民团结精神 |
| Показала, что любой шантаж, попытки устроить внутреннюю смуту обречены на провал | 这表明任何胁迫和制造内乱的企图注定失败 |

翻译特点：
- 理解上下文语境，使用地道表达
- 专有名词保留原文或音译
- 支持反思机制，多轮优化提升准确率

---

## 💻 Web 界面

SubForge 提供现代化的 Web 界面，三步完成字幕处理：

**1. 导入视频** — 拖拽上传或选择文件，支持 MP4 / MKV / AVI / MOV 等格式

**2. 语音转录** — 选择 ASR 引擎和语言，一键开始转录，实时显示进度

**3. 字幕编辑** — 在线编辑原文和译文，支持合并 / 删除 / 重新翻译，导出多格式

界面特点：
- 响应式布局，自适应窗口大小
- 实时进度追踪，WebSocket 推送
- 内联编辑，双击即可修改字幕
- 右键菜单，快捷操作
- LLM 请求日志，可查看每次 API 调用详情

---

## 🚀 快速开始

### 方式一：下载桌面应用（推荐）

前往 [Releases](https://github.com/henry1786580051-lang/SubForge/releases/latest) 页面下载安装包：

| 平台 | 文件 |
|------|------|
| **macOS** | `SubForge-*-arm64.dmg` |
| **Windows** | `SubForge-Setup-*.exe` |

> 💡 macOS 首次打开需在「系统设置 → 隐私与安全性」中点击「仍要打开」。

### 方式二：从源码运行

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

---

## 🤖 LLM 配置

智能断句和智能翻译依赖 LLM。SubForge 使用 OpenAI 兼容接口，支持任意提供商：

| 提供商 | 推荐模型 |
|--------|----------|
| **小米 MiMo** | mimo-v2.5-pro（默认） |
| **DeepSeek** | deepseek-chat |
| **OpenAI** | gpt-4o / gpt-4o-mini |
| **通义千问** | qwen-plus |
| **SiliconFlow** | 多模型可选 |

在「设置」页面选择提供商，填入 API Key 即可。也可使用 LM Studio 等本地模型，完全免费。

---

## 🎙️ ASR 引擎

| 引擎 | 类型 | 说明 |
|------|------|------|
| **Whisper.cpp** | 本地 | 推荐，支持 Metal / CUDA GPU 加速 |
| **Whisper API** | 云端 | OpenAI Whisper API |
| **Faster Whisper** | 本地 | CTranslate2 加速，性能优秀 |
| **剪映 ASR** | 云端 | 剪映同款引擎，中文效果好 |
| **必剪 ASR** | 云端 | B 站必剪引擎，免费使用 |

Whisper.cpp 支持下载不同大小的模型（Tiny 75MB → Large V3 3.1GB），在设置页面一键下载。

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| [Next.js 16](https://nextjs.org/) | 前端框架 |
| [React](https://react.dev/) | UI 组件库 |
| [TypeScript](https://www.typescriptlang.org/) | 类型安全 |
| [Zustand](https://zustand-demo.pmnd.rs/) | 状态管理 |
| [Tailwind CSS](https://tailwindcss.com/) | 样式框架 |
| [FastAPI](https://fastapi.tiangolo.com/) | 后端 API |
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | 本地语音识别 |
| [Electron](https://www.electronjs.org/) | 桌面应用封装 |

---

## 📁 项目结构

```
SubForge/
├── frontend/                  # Next.js 前端
│   └── src/
│       ├── components/        # UI 组件
│       │   ├── VideoPanel.tsx     # 视频导入与播放
│       │   ├── ConfigPanel.tsx    # 转录/翻译配置
│       │   ├── SubtitlePanel.tsx  # 字幕编辑器
│       │   ├── SettingsPanel.tsx  # 全局设置
│       │   └── Sidebar.tsx        # 侧边栏导航
│       ├── store/             # Zustand 状态管理
│       └── lib/               # API 客户端
├── backend/                   # FastAPI 后端
│   └── app/
│       ├── api/               # API 路由（文件/转录/字幕/配置）
│       └── core/              # 任务管理器
└── subforge/                  # Python 核心库
    └── core/
        ├── asr/               # ASR 引擎
        ├── translate/         # 翻译引擎
        ├── llm/               # LLM 调用与日志
        └── subtitle/          # 字幕处理
```

---

## 📄 许可证

[GPL-3.0 License](LICENSE)
