# SubForge CLI

## 安装

```bash
pip install subforge          # CLI 与兼容 GUI 入口
```

免费功能（转录、必应/谷歌翻译）无需任何配置，安装后直接使用。
`subforge-gui`、`subforge gui` 和无参数的 `subforge` 保留兼容 GUI 入口；发布版桌面应用使用 Next.js + FastAPI + pywebview，并通过 Windows/macOS 安装包分发。

---

## 快速开始

```bash
# 语音转字幕（免费）
subforge transcribe video.mp4 --asr whisperx --word-timestamps

# 翻译字幕（免费必应翻译）
subforge subtitle input.srt --translator bing --target-language en

# 全流程：转录 → 优化 → 翻译 → 合成
subforge process video.mp4 --asr whisperx --translator bing --target-language ja

# 给视频加字幕
subforge synthesize video.mp4 -s subtitle.srt --subtitle-mode hard

# 根据字幕生成配音音轨（默认 Edge TTS，无需 API key）
subforge dub subtitle.srt -o dub.wav

# 全流程：视频 → 转录 → 翻译 → 配音视频
subforge process video.mp4 --translator bing --to zh-Hans \
  --dub-only
```

---

## 命令

### `transcribe` — 语音转字幕

将音视频文件转为字幕文件。支持 mp3/wav/mp4/mkv 等格式，视频自动提取音频。

```bash
subforge transcribe <文件> [选项]
```

| 选项 | 说明 |
|------|------|
| `--asr` | ASR 引擎：`whisperx`（Apple Silicon 默认，支持强制对齐）、`whisper-cpp`、`faster-whisper`、`whisper-api` |
| `--language CODE` | 源语言 ISO 639-1 代码，如 `zh` `en` `ja`，或 `auto`（默认） |
| `--word-timestamps` | 输出词级时间戳（配合字幕断句使用） |
| `--whisper-api-key` | Whisper API 密钥（仅 `--asr whisper-api`） |
| `--whisper-api-base` | Whisper API 地址 |
| `--whisper-model` | Whisper 模型名（whisper-api 默认 whisper-1，whisper-cpp 默认 large-v2） |
| `-o PATH` | 输出文件或目录路径 |
| `--format` | 输出格式：`srt`(默认) `ass` `txt` `json` |

---

### `subtitle` — 字幕优化与翻译

处理字幕文件，支持三个步骤：

1. **断句** — 按语义重新分割字幕（LLM）
2. **优化** — 修正 ASR 错误、标点、格式（LLM）
3. **翻译** — 翻译到其他语言（LLM / 必应 / 谷歌）

默认开启优化和断句，翻译默认关闭。指定 `--translator` 或 `--target-language` 自动开启翻译。

```bash
subforge subtitle <字幕文件> [选项]
```

| 选项 | 说明 |
|------|------|
| `--translator` | 翻译服务：`llm`(默认) `bing`(免费) `google`(免费) |
| `--target-language CODE` | 目标语言 BCP 47 代码：`zh-Hans` `en` `ja` `ko` `fr` `de` 等 |
| `--no-optimize` | 跳过优化 |
| `--no-translate` | 跳过翻译 |
| `--no-split` | 跳过断句 |
| `--reflect` | 反思式翻译（仅 LLM，质量更高但更慢） |
| `--layout` | 双语布局：`target-above` `source-above` `target-only` `source-only` |
| `--prompt TEXT` | 自定义提示词（辅助 LLM 优化/翻译） |
| `--api-key` | LLM API 密钥（或设置 `OPENAI_API_KEY` 环境变量） |
| `--api-base` | LLM API 地址（或设置 `OPENAI_BASE_URL` 环境变量） |
| `--model` | LLM 模型名（如 gpt-4o-mini） |

---

### `synthesize` — 字幕合成到视频

将字幕烧录到视频中，支持美观的样式化字幕。

```bash
subforge synthesize <视频> -s <字幕> [选项]
```

| 选项 | 说明 |
|------|------|
| `-s FILE` | **必填**，字幕文件 |
| `--subtitle-mode` | `soft`(默认,嵌入轨道) 或 `hard`(烧录画面) |
| `--quality` | 视频质量：`ultra`(CRF18) `high`(CRF23) `medium`(默认,CRF28) `low`(CRF32) |
| `--layout` | 双语字幕布局 |
| `--style NAME` | 样式预设（运行 `subforge style` 查看） |
| `--style-override JSON` | 内联 JSON 覆盖样式字段，如 `'{"outline_color": "#ff0000"}'` |
| `--render-mode` | 渲染模式：`ass`(默认,描边样式) 或 `rounded`(圆角背景) |
| `--font-file PATH` | 自定义字体文件 (.ttf/.otf) |

#### 字幕样式

SubForge 支持两种渲染模式，让字幕更美观：

**ASS 模式**（默认）— 传统描边/阴影样式，支持自定义字体、颜色、描边宽度：
```bash
# 使用动漫风格预设
subforge synthesize video.mp4 -s sub.srt --subtitle-mode hard --style anime

# 自定义红色描边
subforge synthesize video.mp4 -s sub.srt --subtitle-mode hard \
  --style-override '{"outline_color": "#ff0000", "font_size": 48}'
```

**圆角背景模式** — 现代圆角矩形背景，支持自定义背景色、圆角半径、内边距：
```bash
# 使用圆角背景
subforge synthesize video.mp4 -s sub.srt --subtitle-mode hard --render-mode rounded

# 自定义白字红底
subforge synthesize video.mp4 -s sub.srt --subtitle-mode hard \
  --style-override '{"text_color": "#ffffff", "bg_color": "#ff000099", "corner_radius": 12}'
```

运行 `subforge style` 查看所有预设及其参数。样式选项仅对硬字幕（`--subtitle-mode hard`）生效。

---

### `dub` — 字幕配音

根据字幕时间轴生成配音音轨，可选把音轨写回视频。普通 SRT 可直接使用；多说话人可在字幕文本里写：

```text
[Alice] 你好，今天开始测试。
Bob: This line uses another voice.
```

```bash
# Edge TTS（默认，无需 API key，依赖网络）
subforge dub input.srt \
  --preset edge-cn-female \
  -o output.wav

# SiliconFlow CosyVoice2
subforge dub input.srt \
  --preset siliconflow-cn-female \
  --tts-api-key "$VIDEOCAPTIONER_TTS_API_KEY" \
  -o output.wav

# Gemini TTS
subforge dub input.srt \
  --preset gemini-en-friendly \
  --tts-api-key "$VIDEOCAPTIONER_TTS_API_KEY" \
  -o output.wav

# 多说话人音色映射，并输出视频
subforge dub input.srt --video video.mp4 \
  --speaker-voice Alice=anna \
  --speaker-voice Bob=benjamin \
  -o video_dubbed.mp4
```

| 选项 | 说明 |
|------|------|
| `--preset` | 配音预设：如 `siliconflow-cn-female`、`gemini-en-friendly`、`edge-cn-female` |
| `--tts-api-key` | TTS API key。SiliconFlow/Gemini 需要；Edge TTS 不需要 |
| `--voice` | 默认音色。SiliconFlow 可用 `anna`、`alex`、`benjamin`；Gemini 使用 `Kore`、`Achird`；Edge 可用 `xiaoxiao`、`yunxi` 或完整 voice ID |
| `--speak auto/first/second` | 双语字幕时选择朗读第一行还是第二行 |
| `--speaker-voice NAME=VOICE` | 给字幕中的说话人指定音色，可重复 |
| `--speaker-clone NAME=AUDIO\|TEXT` | SiliconFlow 音色克隆参考音频与对应文本 |
| `--clone-audio` / `--clone-text` | 给默认说话人使用 SiliconFlow 音色克隆；Gemini/Edge 不支持 |
| `--timing balanced/strict/natural/none` | 时间轴策略：默认平衡；`strict` 更贴字幕；`natural` 更保留自然语速 |
| `--adapt-length` | 使用 LLM 缩短明显过长的台词 |
| `--audio-mode replace/mix/duck` | 输出视频时替换原声、混合原声，或压低原声作为背景 |

命令会额外生成 `*.dubbing.json` 报告，记录每句使用的说话人、音色、生成时长、变速倍数和时间轴 warning。

---

### `process` — 全流程处理

一键完成：转录 → 断句 → 优化 → 翻译 → 合成。支持上述所有命令的参数。

```bash
subforge process <音视频文件> [选项]
```

额外选项：

| 选项 | 说明 |
|------|------|
| `--no-synthesize` | 跳过视频合成（只输出字幕） |
| `--dub` | 在转录/处理字幕后生成配音音轨或配音视频 |
| `--dub-only` | 只输出配音结果，跳过字幕烧录/嵌入 |

示例：

```bash
# 英文视频配成中文视频
subforge process talk.mp4 \
  --asr whisperx \
  --translator bing --to zh-Hans \
  --dub-only \
  --timing strict

# 中文视频配成英文视频
subforge process input.mp4 \
  --translator bing --to en \
  --dub-only \
  --preset gemini-en-friendly \
  --tts-api-key "$VIDEOCAPTIONER_TTS_API_KEY"
```

音频文件自动跳过合成步骤。

---

### `download` — 下载在线视频

```bash
subforge download <URL> [-o 目录]
```

支持 YouTube、B站等 yt-dlp 支持的平台。

---

### `style` — 查看字幕样式

```bash
subforge style
```

列出所有可用样式预设及其配置参数，包括 ASS 和圆角背景两种模式。

---

### `config` — 配置管理

```bash
subforge config show              # 查看配置
subforge config set <key> <value> # 设置配置项
subforge config get <key>         # 获取配置项
subforge config path              # 配置文件路径
subforge config init              # 交互式初始化
subforge config init --non-interactive --profile dubbing
subforge config init --print-template
```

---

### `doctor` — 环境诊断

```bash
subforge doctor          # 检查依赖和配置
subforge doctor --json   # Agent/CI 友好的 JSON 输出
```

会检查 Python、FFmpeg/FFprobe、yt-dlp、配置文件、ASR、LLM、翻译和配音关键配置。缺失项会给出对应修复命令。

---

## 配置

配置优先级：命令行参数 > 环境变量 > 配置文件 > 默认值。

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM API 密钥 |
| `OPENAI_BASE_URL` | LLM API 地址 |
| `OPENAI_MODEL` | LLM 模型名 |
| `VIDEOCAPTIONER_DUB_PRESET` | 配音预设 |
| `VIDEOCAPTIONER_TTS_API_KEY` | 配音 TTS API 密钥 |
| `VIDEOCAPTIONER_TTS_API_BASE` | 配音 TTS API 地址 |
| `VIDEOCAPTIONER_TTS_MODEL` | 配音 TTS 模型 |
| `VIDEOCAPTIONER_TTS_VOICE` | 配音默认音色 |
| `VIDEOCAPTIONER_TTS_WORKERS` | 并发 TTS 请求数 |
| `VIDEOCAPTIONER_DUB_TIMING` | 配音时间轴策略 |
| `VIDEOCAPTIONER_DUB_AUDIO_MODE` | 原声处理方式 |
| `VIDEOCAPTIONER_TTS_MAX_SPEED` | 配音最大变速倍数 |
| `VIDEOCAPTIONER_TTS_REWRITE_TOO_LONG` | 是否启用 LLM 缩短过长台词 |

### 配置文件

位置：`~/.config/subforge/config.toml`（macOS/Linux）

推荐先运行：

```bash
subforge config init
subforge doctor
```

非交互环境可以这样初始化：

```bash
subforge config init --non-interactive --profile dubbing \
  --translator bing \
  --timing balanced --audio-mode replace
```

```toml
[llm]
api_key = "sk-xxx"
api_base = "https://api.openai.com/v1"
model = "gpt-4o-mini"

[transcribe]
asr = "whisperx"

[subtitle]
optimize = true
split = true

[translate]
service = "bing"

[dubbing]
preset = "edge-cn-female"
api_key = ""
voice = "xiaoxiao"
timing = "balanced"
audio_mode = "replace"
tts_workers = 5
```

运行 `subforge config show` 查看完整配置项。

---

## 通用选项

| 选项 | 说明 |
|------|------|
| `-v` / `--verbose` | 详细输出 |
| `-q` / `--quiet` | 静默模式，仅输出结果路径（适合管道使用） |
| `--config FILE` | 指定配置文件 |

## 退出码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 参数/配置错误 |
| 3 | 输入文件不存在 |
| 4 | 依赖缺失（FFmpeg 等） |
| 5 | 运行时错误（API 失败等） |
