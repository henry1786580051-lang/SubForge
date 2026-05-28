# 🔥 SubForge

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)

AI 驱动的视频字幕工具 — 语音转录、字幕优化、智能翻译，一站完成。

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
- [转录优化策略](#-转录优化策略)
- [技术栈](#️-技术栈)
- [项目结构](#-项目结构)
- [许可证](#-许可证)

---

## 🌟 核心特性

| 特性 | 说明 |
|------|------|
| **语音降噪** | DeepFilterNet3 深度学习降噪，去除背景噪音保留人声 |
| **智能 VAD** | Silero VAD 预处理，自动跳过静音段，消除幻觉字幕 |
| **智能断句** | LLM 按语义自然分段，而非机械切割，字幕阅读体验流畅 |
| **智能翻译** | 上下文感知翻译，理解整段语境，输出地道表达而非逐词直译 |
| **纠错优化** | 自动去除语气词、修正错别字、规范标点格式 |
| **双语字幕** | 原文 + 译文对照输出，支持 SRT / VTT / ASS / TXT / JSON |
| **多引擎 ASR** | Whisper.cpp（GPU 加速）、Whisper API、Faster Whisper |
| **Web 界面** | 浏览器直接使用，拖拽上传、实时进度、在线编辑字幕 |

---

## 🧠 智能断句

语音识别输出的原始文本通常是冗长的连续句子，直接作为字幕难以阅读。SubForge 使用 LLM 按语义自然断点重新分段。

以「[2026 Lexus ES 350h 试驾](https://www.youtube.com/watch?v=mozncOwkny4)」视频为例：

**ASR 原始输出**（Whisper 转录）：
```
As specced, this gets about 48 miles per gallon on the highway, about 44 in the city, 46 combined,
which is an amazing fuel economy, 244 net combined horsepower, and a completely new
interior and exterior. Let's walk you guys around it, talk about what it's been like
to spend the first 30, 40 minutes behind the wheel of this car, and give you guys
some first driving impressions.
```

**LLM 智能断句后**：
```
官方数据显示，它高速巡航的油耗大概在百公里4.9升左右。
市区大概百公里5.4升，综合下来5.1升左右。
这油耗表现相当惊人。
综合马力有244匹。
内外饰都是全新的设计。
我带大家先绕车看一圈。
刚才开了三四十分钟，跟大家聊聊我的感受。
分享一下初驾感受。
```

> 完整示例：[`examples/lexus_original.srt`](examples/lexus_original.srt)（ASR 原始输出）→ [`examples/lexus_processed.srt`](examples/lexus_processed.srt)（断句 + 翻译后）

断句效果：
- 冗长的连续文本被拆分为短小精悍的字幕段
- 每段保持完整语义，不在句子中间断开
- 自动补全标点符号，提升可读性

---

## 🌐 智能翻译

SubForge 的 LLM 翻译结合上下文理解整段内容，输出自然流畅的中文，而非逐词直译：

| ASR 原始英文 | LLM 中文翻译 |
|------|----------|
| Today we are driving the all-new 2026 Lexus ES 350h. | 今天来试试2026新款雷克萨斯ES 350h。 |
| This is the premium front-wheel drive. | 这回选的是顶配的前驱版。 |
| As tested, this is about $53,000. | 这台测试车的售价，大概在五万三千美金左右，折合人民币差不多三十八万。 |
| We have the palomino interior, which looks very nice on here. | 内饰配的是帕洛米诺棕，质感确实很棒。 |
| Of course this is a fully redesigned inside and out. | 而且，这次是彻头彻尾的换代，里里外外都是新的。 |
| It's available as an EV or as this 350h hybrid. | 它有两种动力可选：纯电版，以及像咱们这台350h的混动版。 |

翻译特点：
- 理解上下文语境，使用地道中文表达
- 车型名称「Lexus ES」「350h」保留原文
- 单位自动换算（英里/加仑 → 百公里油耗，美元 → 人民币）
- 口语化表达，匹配视频博主的轻松语气

### 反思翻译模式

SubForge 支持「反思翻译」模式（Inspired by Andrew Ng's Reflection Pattern），通过三阶段流程提升译文质量：

1. **初译** — LLM 完成第一遍翻译
2. **反思** — 自动检测机翻痕迹：语序生硬、用词机械、文化不匹配、语域不当
3. **重写** — 基于反思结果，输出母语者表达习惯的自然译文

以「[2026 Lexus ES 350h 试驾](https://www.youtube.com/watch?v=mozncOwkny4)」为例，反思模式的实际效果：

| 阶段 | 内容 |
|------|------|
| **初译** | 今天我们驾驶的是全新2026款雷克萨斯ES 350h。 |
| **反思** | "今天我们驾驶的是"照搬英文语序，中文博主会说"今天来试试"。"全新2026款"像新闻稿，口语中说"2026新款"。 |
| **重写** | 今天来试试2026新款雷克萨斯ES 350h。 |

| 阶段 | 内容 |
|------|------|
| **初译** | 测试车的价格大约是53,000美元。 |
| **反思** | "测试车的价格大约是"过于正式。中文观众习惯"万"为单位，需换算。博主语气应更口语化。 |
| **重写** | 这台测试车的售价，大概在五万三千美金左右，折合人民币差不多三十八万。 |

开启后，翻译耗时会略有增加，但译文质量显著提升，尤其适合对表达地道程度要求较高的内容。

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

Whisper.cpp 支持下载不同大小的模型（Tiny 75MB → Large V3 3.1GB），在设置页面一键下载。

---

## 🎯 转录优化策略

SubForge 采用三层音频预处理管线，在 ASR 转录前大幅降低噪音和静音干扰：

```
原始音频 → DeepFilterNet3 降噪 → Silero VAD 切割 → 只转录语音片段 → 合并字幕
```

### 第一层：DeepFilterNet3 语音降噪

[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) 是德国凯泽斯劳滕大学开发的深度学习语音增强模型。它不是简单的滤波器，而是真正"听懂"语音后再去除噪音。

对驾驶视频、户外评测等有持续背景噪音的场景效果显著：

| 场景 | 降噪前 | 降噪后 |
|------|--------|--------|
| 车内评测 | 引擎声、风声干扰 | 清晰人声 |
| 户外采访 | 环境嘈杂 | 突出对话 |
| 咖啡厅访谈 | 背景音乐、杯碟声 | 干净语音 |

### 第二层：Silero VAD 语音检测

[Silero VAD](https://github.com/snakers4/silero-vad) 是轻量级语音活动检测模型（~2MB），在转录前精确识别语音段落，跳过静音区域。

效果对比（21 分钟 Lexus ES 350h 试驾视频）：

| 指标 | 无 VAD | + DeepFilterNet | + Silero VAD |
|------|--------|-----------------|--------------|
| 字幕段数 | 245 | 241 | 314 |
| 覆盖率 | 99.1% | 94.3% | 79.7% |
| 静音间隙 | 0 | 5 | 46 |
| 幻觉字幕 | 大量 | 减少 | 基本消除 |
| 转录速度 | 1x | 1x | ~3x |

### 质量保障：Content Integrity Score (CIS)

为防止参数优化过程中误吞语音内容，SubForge 使用 **CIS（内容完整性评分）** 作为安全护栏：

```
CIS = 当前参数语音时长 / 宽松基准语音时长（threshold=0.2）

CIS > 0.90  →  ✅ 安全，内容完整
CIS 0.85~0.90  →  ⚠️ 警告，需关注
CIS < 0.85  →  🛑 停止，正在丢失内容
```

该指标不需要人工标注 ground truth，可跨视频横向比较，确保参数在迭代优化中不会悄悄朝错误方向漂移。

### 参数优化工具

项目内置 `scripts/vad_benchmark.py` 参数调优工具：

```bash
# 测试单个视频
uv run python scripts/vad_benchmark.py --video /path/to/video.mp4

# 参数敏感度分析
uv run python scripts/vad_benchmark.py --sensitivity

# 网格搜索最优参数
uv run python scripts/vad_benchmark.py --grid

# 查看历史结果与趋势
uv run python scripts/vad_benchmark.py --report
```

报告自动输出 CIS、Gap Density（间隙密度）、段长分布（P5/P50/P95）、趋势监控等指标，并对异常值发出告警。

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
