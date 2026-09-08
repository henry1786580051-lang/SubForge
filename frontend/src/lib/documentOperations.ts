import { useAppStore } from "../store/appStore";
import type { SubtitleFile } from "./api";
import { hasUnsavedSubtitles } from "./subtitleEdits";

/** Every subtitle import commits its path, contents and baseline together. */
export async function importSubtitleDocument(load: () => Promise<SubtitleFile>) {
  const before = useAppStore.getState();
  if (before.isProcessing) throw new Error("请先等待当前任务结束或取消任务，再导入字幕。");
  if (before.subtitleSaving || before.documentLoading) throw new Error("文件正在保存或读取，请稍后再导入。");
  if (hasUnsavedSubtitles(before)) throw new Error("请先保存当前字幕的修改，再导入新字幕。");
  useAppStore.setState({ documentLoading: true, currentTaskId: null });
  try {
    const loaded = await load();
    if (useAppStore.getState().subtitleDocumentRevision !== before.subtitleDocumentRevision) return;
    useAppStore.setState({
      subtitleFile: loaded.file_path, subtitles: loaded.segments, savedSubtitles: loaded.segments,
      subtitleDocumentRevision: before.subtitleDocumentRevision + 1,
      subtitleHistory: [], selectedIds: new Set(), taskStatus: "idle", taskProgress: 0,
      taskMessage: "", taskAttention: null,
    });
  } finally { useAppStore.setState({ documentLoading: false }); }
}
