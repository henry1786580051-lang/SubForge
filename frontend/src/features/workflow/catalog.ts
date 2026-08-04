import type { WorkflowStep } from "@/store/appStore";

export const STEP_META: Record<
  WorkflowStep,
  { eyebrow: string; title: string; description: string; icon: string }
> = {
  import: {
    eyebrow: "01 / 导入素材",
    title: "开始新的字幕任务",
    description: "添加视频、音频或已有字幕，系统将自动读取文件信息并检查可用音轨。",
    icon: "solar:inbox-in-bold-duotone",
  },
  transcribe: {
    eyebrow: "02 / 语音转录",
    title: "生成带时间轴的字幕",
    description: "选择识别模型和源语言，实时查看转录进度与时间轴质量。",
    icon: "solar:microphone-3-bold-duotone",
  },
  subtitle: {
    eyebrow: "03 / 字幕处理",
    title: "断句、翻译与审校",
    description: "优化字幕分段并生成目标语言译文，完成后可直接检查和导出。",
    icon: "solar:subtitle-bold-duotone",
  },
};

export const SOURCE_LANGUAGES = [
  ["auto", "自动"],
  ["en", "英文"],
  ["zh", "中文"],
  ["ja", "日文"],
  ["ko", "韩文"],
  ["fr", "法语"],
  ["de", "德语"],
  ["es", "西班牙语"],
  ["pt", "葡萄牙语"],
  ["ru", "俄语"],
] as const;

export const TARGET_LANGUAGES = [
  ["chinese", "中文"],
  ["english", "英文"],
  ["japanese", "日文"],
  ["korean", "韩文"],
  ["french", "法语"],
  ["german", "德语"],
  ["spanish", "西班牙语"],
  ["portuguese", "葡萄牙语"],
  ["russian", "俄语"],
] as const;

export const TRANSLATORS = [
  ["llm", "LLM"],
  ["bing", "Microsoft Azure"],
  ["google", "Google"],
  ["deeplx", "DeepLX"],
] as const;

export const ASR_ENGINES = [
  {
    id: "whisperx",
    name: "WhisperX",
    desc: "高精度识别 · 支持词级时间轴",
    icon: "solar:bolt-bold-duotone",
  },
  {
    id: "whisper_cpp",
    name: "Whisper.cpp",
    desc: "轻量本地识别 · 支持 Metal 与 CPU",
    icon: "solar:cpu-bolt-bold-duotone",
  },
  {
    id: "faster_whisper",
    name: "FasterWhisper",
    desc: "高速本地识别 · 适合 NVIDIA 显卡",
    icon: "solar:cpu-bold-duotone",
  },
  {
    id: "whisper_api",
    name: "Whisper API",
    desc: "云端识别 · 无需下载本地模型",
    icon: "solar:cloud-bold-duotone",
  },
] as const;

export const TRANSCRIBE_STAGES = [
  "提取音轨",
  "优化音频",
  "识别语音",
  "对齐时间轴",
  "区分说话人",
  "检查时间轴",
  "保存字幕",
] as const;
