import { describe, expect, it } from "vitest";
import { downloadLabel } from "./downloadProgress";
import type { DownloadProgress } from "./api";
const data: DownloadProgress = { phase: "downloading", downloaded_bytes: 1048576, total_bytes: 10485760, speed_bps: 1048576, eta_seconds: 120, progress: 10 };
describe("download status", () => {
  it("shows bytes, speed and a rounded download estimate", () => {
    expect(downloadLabel({ download: data, message: "" })).toBe("1.0 MB / 10.0 MB · 1.0 MB/s · 预计下载还需约 2 分钟");
  });
  it("does not show a stale ETA while stalled", () => {
    expect(downloadLabel({ download: { ...data, phase: "waiting" }, message: "" })).toBe("1.0 MB / 10.0 MB · 等待网络响应…");
  });
  it("keeps verification distinct from a countdown", () => {
    expect(downloadLabel({ download: null, message: "正在校验模型文件…" })).toBe("正在校验模型文件…");
  });
  it("does not invent an estimate without size", () => {
    expect(downloadLabel({ download: { ...data, total_bytes: null, eta_seconds: null }, message: "" })).toContain("总大小未知");
  });
});
