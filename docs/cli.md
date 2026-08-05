# SubForge CLI

命令行版本提供与当前产品一致的核心能力：媒体转录、字幕断句、优化、翻译、模型下载和环境诊断。它只生成字幕文件，不修改或重新编码源视频。

## 快速开始

```bash
uv run subforge doctor
uv run subforge transcribe input.mp4 --asr whisperx --language auto --word-timestamps
uv run subforge subtitle input.srt --translator llm --target-language zh-Hans
uv run subforge process input.mp4 --asr whisperx --translator llm --target-language zh-Hans
```

## 命令

### `transcribe`

将音频或视频转为字幕。支持 `whisperx`、`whisper-cpp`、`faster-whisper` 和 `whisper-api`。

```bash
subforge transcribe input.mp4 -o output.srt --asr whisperx --language auto
```

### `subtitle`

对已有字幕执行语义断句、ASR 文本优化与翻译。

```bash
subforge subtitle input.srt -o output.srt \
  --translator llm --target-language zh-Hans \
  --batch-size 20 --thread-num 20
```

双语布局可选 `target-above`、`source-above`、`target-only`、`source-only`。

### `process`

依次执行转录与字幕处理，输出原始 SRT 和处理后的 SRT，不修改输入媒体。

```bash
subforge process input.mp4 --asr whisperx --translator llm --target-language zh-Hans
```

### `download`

使用 yt-dlp 下载其支持的网站媒体。

```bash
subforge download "https://example.com/video" -o ./downloads
```

### `config`

```bash
subforge config init
subforge config show
subforge config set llm.model model-name
subforge config path
```

配置优先级为：命令行参数、环境变量、用户配置文件、内置默认值。

### `doctor`

检查 Python、FFmpeg、FFprobe、yt-dlp、ASR 与翻译配置。

```bash
subforge doctor
subforge doctor --json
```

FFmpeg 仍用于提取音轨、统一采样率和声道；FFprobe 用于读取媒体时长和音轨信息。当前版本不提供字幕烧录或视频合成。
