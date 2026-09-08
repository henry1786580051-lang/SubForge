import type { AsrModelInfo } from "./api";

export function modelPresentation(model: Pick<AsrModelInfo, "name" | "id" | "downloaded" | "state">) {
  const raw = (model.name || model.id.split(/[\\/]/).pop() || model.id).trim();
  const match = raw.match(/^(?:(?:mlx|ggml|whisper|faster-whisper)[ _-]+)?(tiny|base|small|medium|large)(?=$|[ ._-])(?:[ _-]+(v[123]))?(?:[ _-]+(turbo))?(.*)$/i);
  const family = match?.[1].toLowerCase();
  const title = match ? `Whisper ${family![0].toUpperCase()}${family!.slice(1)}${match[2] ? ` ${match[2].toLowerCase()}` : ""}${match[3] ? " Turbo" : ""}` : raw;
  const variant = match?.[4].replace(/^[ _.-]+/, "").trim() || "";
  const hint = match?.[3] ? "加速版本 · 兼顾速度与精度" : family === "tiny" || family === "base" ? "速度优先 · 适合快速预览" : family === "small" || family === "medium" ? "兼顾速度与识别效果" : family === "large" ? "精度优先 · 计算需求较高" : "自定义模型 · 查看详情了解配置";
  const status = model.downloaded ? "本地可用" : model.state === "on_demand" ? "首次使用自动下载" : "尚未下载";
  return { title, variant, hint, status };
}
