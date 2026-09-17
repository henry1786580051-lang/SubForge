import type { TaskInfo } from "./api";

const bytes = (value: number) => {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
};

export function downloadLabel(task: Pick<TaskInfo, "download" | "message"> | null): string {
  const data = task?.download;
  if (!data) return task?.message || "正在连接下载服务…";
  const size = data.total_bytes ? `${bytes(data.downloaded_bytes)} / ${bytes(data.total_bytes)}` : `已下载 ${bytes(data.downloaded_bytes)}`;
  if (data.phase === "retrying") return `${size} · 正在重新传输…`;
  if (data.phase === "waiting") return `${size} · 等待网络响应…`;
  const rate = data.speed_bps ? `${bytes(data.speed_bps)}/s` : null;
  const eta = data.eta_seconds;
  const range = data.eta_range_seconds;
  const time = range && range[1] >= 60
    ? `预计下载还需约 ${Math.max(1, Math.ceil(range[0] / 60))}–${Math.max(1, Math.ceil(range[1] / 60))} 分钟`
    : eta == null ? (data.total_bytes ? "正在估算…" : "总大小未知")
    : eta < 60 ? "预计下载还需不到 1 分钟"
    : eta < 3600 ? `预计下载还需约 ${Math.ceil(eta / 60)} 分钟`
    : `预计下载还需约 ${(Math.ceil(eta / 600) / 6).toFixed(1)} 小时`;
  return [size, rate, time].filter(Boolean).join(" · ");
}
