interface LlmProviderDefinition {
  id: string;
  name: string;
  baseUrl: string;
  groupedModels?: boolean;
}

export const LLM_PROVIDERS: ReadonlyArray<LlmProviderDefinition> = [
  { id: "openai", name: "OpenAI", baseUrl: "https://api.openai.com/v1" },
  { id: "nvidia", name: "NVIDIA", baseUrl: "https://integrate.api.nvidia.com/v1", groupedModels: true },
  { id: "deepseek", name: "DeepSeek", baseUrl: "https://api.deepseek.com" },
  { id: "mimo", name: "小米 MiMo", baseUrl: "https://token-plan-cn.xiaomimimo.com/v1" },
  { id: "qwen", name: "通义千问", baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1" },
  { id: "zhipu", name: "智谱 GLM", baseUrl: "https://open.bigmodel.cn/api/paas/v4" },
  { id: "moonshot", name: "月之暗面", baseUrl: "https://api.moonshot.cn/v1" },
  { id: "baichuan", name: "百川智能", baseUrl: "https://api.baichuan-ai.com/v1" },
  { id: "yi", name: "零一万物", baseUrl: "https://api.lingyiwanwu.com/v1" },
  { id: "minimax", name: "MiniMax", baseUrl: "https://api.minimaxi.com/anthropic" },
  { id: "siliconflow", name: "SiliconFlow", baseUrl: "https://api.siliconflow.cn/v1" },
  { id: "openrouter", name: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1" },
  { id: "custom", name: "自定义", baseUrl: "" },
];

export type SettingsView = "general" | "llm" | "asr" | "subtitle" | "files";

export const ENGLISH_LENGTH_PRESETS = [
  { label: "紧凑", value: 14, description: "目标 14 词 · 最多 18 词" },
  { label: "均衡", value: 18, description: "目标 18 词 · 最多 22 词" },
  { label: "宽松", value: 22, description: "目标 22 词 · 最多 26 词" },
] as const;

export const SETTINGS_VIEWS: ReadonlyArray<{
  id: SettingsView;
  label: string;
  description: string;
  icon: string;
}> = [
  { id: "general", label: "通用", description: "外观、快捷键与使用支持", icon: "solar:settings-linear" },
  { id: "llm", label: "LLM 服务", description: "服务商、模型与性能", icon: "solar:server-square-cloud-linear" },
  { id: "asr", label: "语音识别", description: "引擎、模型与时间轴", icon: "solar:microphone-3-linear" },
  { id: "subtitle", label: "字幕处理", description: "翻译、断句与输出", icon: "solar:subtitles-linear" },
  { id: "files", label: "文件与存储", description: "工作目录", icon: "solar:folder-with-files-linear" },
];

export const WHISPER_CPP_MODELS = [
  { id: "tiny", name: "Tiny", size: "75MB", desc: "39M 参数，速度最快，适合快速预览，多语言能力弱" },
  { id: "base", name: "Base", size: "142MB", desc: "74M 参数，速度很快，英文表现尚可，其他语言一般" },
  { id: "small", name: "Small", size: "466MB", desc: "244M 参数，速度与质量较平衡，多语言能力明显提升" },
  { id: "medium", name: "Medium", size: "1.5GB", desc: "769M 参数，高准确率，中日韩等非英语语言推荐起步" },
  { id: "large-v1", name: "Large V1", size: "3.1GB", desc: "1550M 参数，初代旗舰，多语言表现优秀" },
  { id: "large-v2", name: "Large V2", size: "3.1GB", desc: "1550M 参数，训练数据更多，比 V1 更稳定可靠" },
  { id: "large-v3", name: "Large V3", size: "3.1GB", desc: "1550M 参数，最新架构，幻觉更少，推荐高质量转录" },
] as const;

export const FASTER_WHISPER_MODELS = [
  { id: "tiny", name: "Tiny", size: "75MB", desc: "CTranslate2 加速，速度极快，适合实时或低配设备" },
  { id: "base", name: "Base", size: "148MB", desc: "CTranslate2 加速，比原版快 4 倍，日常英文够用" },
  { id: "small", name: "Small", size: "496MB", desc: "CTranslate2 加速，性价比最高，多语言可用" },
  { id: "medium", name: "Medium", size: "1.5GB", desc: "CTranslate2 加速，非英语语言推荐，质量接近 Large" },
  { id: "large-v1", name: "Large V1", size: "3.1GB", desc: "CTranslate2 加速，初代旗舰量化版" },
  { id: "large-v2", name: "Large V2", size: "3.1GB", desc: "CTranslate2 加速，比 V1 训练更充分，更少出错" },
  { id: "large-v3", name: "Large V3", size: "3.1GB", desc: "CTranslate2 加速，最佳质量，专业字幕制作首选" },
  { id: "large-v3-turbo", name: "Large V3 Turbo", size: "1.7GB", desc: "V3 蒸馏版，速度提升 8 倍，质量略低于 V3" },
] as const;

export const MLX_WHISPER_MODELS = FASTER_WHISPER_MODELS.map((model) => ({
  ...model,
  onDemand: true,
}));
