# ASR 配置指南

语音识别（ASR）配置详解。

## 支持的 ASR 引擎

| 引擎 | 特点 | 推荐场景 |
|------|------|---------|
| **FasterWhisper** | 准确度高，支持GPU | 推荐使用 |
| **WhisperCpp** | 轻量级 | CPU环境 |
| **Whisper API** | 云端服务 | 无需本地模型 |

FasterWhisper 使用 CTranslate2 模型目录（包含 `config.json`、`model.bin` 和 `tokenizer.json`）。WhisperCpp 使用单个 `ggml-*.bin` 文件；两个格式不能互换。源码运行时请安装 `faster-whisper` 可选依赖：`uv sync --extra faster-whisper`。
| **B接口/J接口** | 免费在线 | 快速测试 |

## 多人语音识别

说话人分离由 pyannote Community-1 独立完成，可与 Whisper.cpp、FasterWhisper、Whisper API 以及跨平台 WhisperX 组合使用。源码运行时请安装 `diarization` 可选依赖：

```bash
uv sync --extra diarization
```

在界面中选择“双人”“自动人数”或固定 2–10 人，填写具有读取权限的 Hugging Face Token，并先下载 Community-1 模型。说话人标签只作为内部元数据参与断句、翻译和配音，不会自动写入最终字幕文本。

## 模型下载

待补充...

## 配置参数

待补充...

---

相关文档：
- [快速开始](/guide/getting-started)
- [LLM 配置](/config/llm)
