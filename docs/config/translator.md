# 翻译配置指南

字幕翻译配置详解。

## 支持的翻译服务

| 服务 | 特点 | 推荐场景 |
|------|------|---------|
| **LLM 翻译** | 质量最好 | 追求质量 |
| **Microsoft Azure Translator** | 官方 API、批量稳定 | 快速机器翻译 |
| **Google 翻译** | 质量好 | 英语翻译 |
| **DeepLX** | 专业翻译 | 自建服务 |

## 配置方法

在应用的“设置 > 字幕处理 > 翻译服务”中选择 Microsoft Azure Translator，填写：

- Azure Translator API Key
- 资源区域（区域或多服务资源需要；全球单服务资源可留空）
- 服务终结点（默认 `https://api.cognitive.microsofttranslator.com`）

保存后使用“测试连接”确认配置。代码中的历史服务标识仍为 `bing`，用于兼容旧任务和设置文件，实际请求已使用微软官方 Translator v3 API。

## 支持的目标语言

待补充...

---

相关文档：
- [LLM 配置](/config/llm)
- [快速开始](/guide/getting-started)
