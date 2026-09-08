import { beforeEach, expect, it, vi } from "vitest";
import { useAppStore } from "../store/appStore";
import { importSubtitleDocument } from "./documentOperations";
const segment = { id: 1, start: "00:00:00.000", end: "00:00:02.000", text: "Original", translated: "" };
beforeEach(() => {
  useAppStore.setState({ isProcessing: false, documentLoading: false, subtitleSaving: false, currentTaskId: null });
  useAppStore.getState().setSubtitleFile("old.srt");
  useAppStore.getState().setSubtitles([segment]);
});
it("blocks import during a task without even reading the new file", async () => {
  useAppStore.setState({ isProcessing: true, currentTaskId: "running" });
  const load = vi.fn();
  await expect(importSubtitleDocument(load)).rejects.toThrow("当前任务");
  expect(load).not.toHaveBeenCalled();
  expect(useAppStore.getState().subtitleFile).toBe("old.srt");
});
it("keeps path and contents paired until reading succeeds", async () => {
  let resolve!: (value: { file_path: string; segments: typeof segment[]; count: number; format: string }) => void;
  const pending = new Promise<{ file_path: string; segments: typeof segment[]; count: number; format: string }>((done) => { resolve = done; });
  useAppStore.setState({ currentTaskId: "previous-completed-task" });
  const importing = importSubtitleDocument(() => pending);
  expect(useAppStore.getState().subtitleFile).toBe("old.srt");
  expect(useAppStore.getState().currentTaskId).toBeNull();
  useAppStore.getState().updateSubtitle(1, "text", "blocked during import");
  expect(useAppStore.getState().subtitles[0].text).toBe("Original");
  await expect(importSubtitleDocument(vi.fn())).rejects.toThrow("正在保存或读取");
  resolve({ file_path: "new.srt", segments: [{ ...segment, text: "New" }], count: 1, format: "srt" });
  await importing;
  expect(useAppStore.getState().subtitleFile).toBe("new.srt");
  expect(useAppStore.getState().subtitles[0].text).toBe("New");
});
it("preserves the original document if reading fails", async () => {
  await expect(importSubtitleDocument(async () => { throw new Error("invalid"); })).rejects.toThrow("invalid");
  expect(useAppStore.getState().subtitleFile).toBe("old.srt");
  expect(useAppStore.getState().subtitles).toEqual([segment]);
  expect(useAppStore.getState().documentLoading).toBe(false);
});
it("protects unsaved edits and an in-flight save across page changes", async () => {
  useAppStore.getState().updateSubtitle(1, "text", "Edited");
  await expect(importSubtitleDocument(vi.fn())).rejects.toThrow("请先保存");
  useAppStore.setState({ subtitleSaving: true });
  await expect(importSubtitleDocument(vi.fn())).rejects.toThrow("正在保存或读取");
});
