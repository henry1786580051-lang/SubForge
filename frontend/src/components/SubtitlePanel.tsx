"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useAppStore } from "@/store/appStore";
import { subtitleApi, subtitlesApi, filesApi, configApi } from "@/lib/api";

export function SubtitlePanel({
  focusRequest = null,
  showPrompt = true,
  showTranslateActions = true,
}: {
  focusRequest?: { id: number; token: number } | null;
  showPrompt?: boolean;
  showTranslateActions?: boolean;
} = {}) {
  const { subtitles, setSubtitles, updateSubtitle, selectedIds, toggleSelect, selectAll, deselectAll, subtitleFile, config, setError } = useAppStore();
  const [editingCell, setEditingCell] = useState<{ id: number; field: "text" | "translated" } | null>(null);
  const [editValue, setEditValue] = useState("");
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; id: number | null } | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [focusedRowId, setFocusedRowId] = useState<number | null>(null);
  const promptDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tableScrollRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef(new Map<number, HTMLTableRowElement>());
  // TanStack Virtual intentionally exposes mutable measurement helpers.
  // eslint-disable-next-line react-hooks/incompatible-library
  const rowVirtualizer = useVirtualizer({
    count: subtitles.length,
    getScrollElement: () => tableScrollRef.current,
    estimateSize: () => 64,
    getItemKey: (index) => subtitles[index]?.id ?? index,
    measureElement: (element) => element.getBoundingClientRect().height,
    overscan: 8,
  });
  useEffect(() => () => { if (promptDebounceRef.current) clearTimeout(promptDebounceRef.current); }, []);

  useEffect(() => {
    if (!focusRequest) return;
    const rowIndex = subtitles.findIndex((subtitle) => subtitle.id === focusRequest.id);
    if (rowIndex < 0) return;
    rowVirtualizer.scrollToIndex(rowIndex, { align: "center" });
    setFocusedRowId(focusRequest.id);
    let attempts = 0;
    let focusFrame = 0;
    const focusRow = () => {
      const row = rowRefs.current.get(focusRequest.id);
      if (row) {
        row.focus({ preventScroll: true });
        return;
      }
      attempts += 1;
      if (attempts < 12) focusFrame = window.requestAnimationFrame(focusRow);
    };
    focusFrame = window.requestAnimationFrame(focusRow);
    const timer = window.setTimeout(() => setFocusedRowId(null), 2200);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.clearTimeout(timer);
    };
  }, [focusRequest, rowVirtualizer, subtitles]);

  const startEdit = useCallback((id: number, field: "text" | "translated", value: string) => { setEditingCell({ id, field }); setEditValue(value); }, []);
  const commitEdit = useCallback(() => { if (!editingCell) return; updateSubtitle(editingCell.id, editingCell.field, editValue); setEditingCell(null); }, [editingCell, editValue, updateSubtitle]);

  const deleteSelected = useCallback(() => {
    const ids = useAppStore.getState().selectedIds;
    if (ids.size === 0) return;
    const updated = subtitles
      .filter((s) => !ids.has(s.id))
      .map((s, i) => ({ ...s, id: i + 1 }));
    setSubtitles(updated);
    deselectAll();
  }, [subtitles, setSubtitles, deselectAll]);

  const mergeSelected = useCallback(() => {
    const ids = useAppStore.getState().selectedIds;
    if (ids.size < 2) return;
    const selected = subtitles.filter((s) => ids.has(s.id));
    const speakers = new Set(selected.map((s) => s.speaker || ""));
    if (speakers.size > 1) {
      setError("不能合并不同说话人的字幕");
      return;
    }
    const others = subtitles.filter((s) => !ids.has(s.id));
    const joinText = (values: string[]) => values.filter(Boolean).join(" ").replace(/\s+([,.;:!?，。！？；：、])/g, "$1");
    const merged = {
      id: selected[0].id,
      start: selected[0].start,
      end: selected[selected.length - 1].end,
      text: joinText(selected.map((s) => s.text)),
      translated: joinText(selected.map((s) => s.translated)),
      speaker: selected[0].speaker || "",
    };
    const result = [...others, merged].sort((a, b) => a.id - b.id).map((s, i) => ({ ...s, id: i + 1 }));
    setSubtitles(result);
    deselectAll();
  }, [subtitles, setSubtitles, deselectAll, setError]);

  const translateAll = useCallback(async () => {
    if (!subtitleFile) return;
    setIsTranslating(true);
    const store = useAppStore.getState();
    store.setError(null);
    store.setIsProcessing(true);
    store.setTaskState(0, "Starting translation...", "running");
    try {
      const result = await subtitleApi.start({
        subtitle_file: subtitleFile,
        target_language: config.targetLanguage,
        translator: config.translator,
        llm_model: config.llmModel,
        need_optimize: false,
        need_translate: true,
        need_reflect: config.needReflect,
        custom_prompt: config.customPrompt,
      });
      store.setCurrentTaskId(result.task_id);
    } catch (err) {
      store.setError(err instanceof Error ? err.message : "Translation failed");
      store.setTaskState(0, "", "idle");
      store.setIsProcessing(false);
    } finally { setIsTranslating(false); }
  }, [subtitleFile, config]);

  const retranslateAll = useCallback(async () => {
    if (!subtitleFile) return;
    setIsTranslating(true);
    const store = useAppStore.getState();
    store.setError(null);
    store.setIsProcessing(true);
    store.setTaskState(0, "Starting translation...", "running");
    try {
      const result = await subtitleApi.start({
        subtitle_file: subtitleFile,
        target_language: config.targetLanguage,
        translator: config.translator,
        llm_model: config.llmModel,
        need_optimize: config.needOptimize,
        need_translate: true,
        need_reflect: config.needReflect,
        custom_prompt: config.customPrompt,
      });
      store.setCurrentTaskId(result.task_id);
    } catch (err) {
      store.setError(err instanceof Error ? err.message : "Translation failed");
      store.setTaskState(0, "", "idle");
      store.setIsProcessing(false);
    } finally { setIsTranslating(false); }
  }, [subtitleFile, config]);

  const [exportFormat, setExportFormat] = useState<"srt" | "vtt" | "ass" | "txt" | "json">("srt");
  const [exportMode, setExportMode] = useState<"original" | "translated" | "bilingual">("bilingual");
  const [showExportMenu, setShowExportMenu] = useState(false);
  const getExportFilename = useCallback((format: string) => {
    const source = subtitleFile?.split(/[\\/]/).pop() || "subtitles.srt";
    return source.replace(/\.[^.]+$/, `.${format}`);
  }, [subtitleFile]);
  const handleExport = useCallback(async (format?: string, mode?: string) => {
    if (!subtitleFile) return;
    const f = format || exportFormat;
    const m = mode || exportMode;
    const defaultName = getExportFilename(f);
    try {
      // pywebview: use POST endpoint + native save dialog
      if (typeof window !== "undefined" && "pywebview" in window) {
        const pywin = window as unknown as { pywebview?: { api?: { save_file?: (b: string, n: string) => Promise<{ ok: boolean; path?: string; error?: string }> } } };
        // Wait for API to be ready (max 3s)
        if (!pywin.pywebview?.api?.save_file) {
          for (let i = 0; i < 30; i++) {
            await new Promise(r => setTimeout(r, 100));
            if (pywin.pywebview?.api?.save_file) break;
          }
        }
        if (!pywin.pywebview?.api?.save_file) {
          useAppStore.getState().setError("原生保存 API 未就绪，请重启应用");
          return;
        }
        const blob = await subtitlesApi.exportPost(subtitles, f, m, defaultName);
        const reader = new FileReader();
        reader.onerror = () => { useAppStore.getState().setError("文件读取失败"); };
        reader.onload = async () => {
          const base64 = (reader.result as string).split(",")[1];
          try {
            const result = await pywin.pywebview!.api!.save_file!(base64, defaultName);
            if (result && !result.ok && result.error) {
              useAppStore.getState().setError(`保存失败: ${result.error}`);
            }
          } catch (e) {
            console.error("pywebview save_file failed:", e);
            useAppStore.getState().setError("保存失败: " + String(e));
          }
        };
        reader.readAsDataURL(blob);
      } else {
        // Browser: use blob download
        const blob = await subtitlesApi.exportPost(subtitles, f, m, defaultName);
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = defaultName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
      }
      setShowExportMenu(false);
    } catch (err) {
      useAppStore.getState().setError(err instanceof Error ? err.message : "Export failed");
    }
  }, [subtitleFile, subtitles, exportFormat, exportMode, getExportFilename]);
  const handleSave = useCallback(async () => { if (!subtitleFile || subtitles.length === 0) return; try { await subtitlesApi.save(subtitleFile, subtitles); } catch (err) { useAppStore.getState().setError(err instanceof Error ? err.message : "Save failed"); } }, [subtitleFile, subtitles]);

  const handleContextMenu = useCallback((e: React.MouseEvent, id: number) => { e.preventDefault(); e.stopPropagation(); if (!selectedIds.has(id)) toggleSelect(id); setContextMenu({ x: e.clientX, y: e.clientY, id }); }, [selectedIds, toggleSelect]);

  useEffect(() => { const close = () => { setContextMenu(null); setShowExportMenu(false); }; window.addEventListener("click", close); return () => window.removeEventListener("click", close); }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (editingCell) return;
      if (e.key === "Delete") { e.preventDefault(); deleteSelected(); }
      if ((e.metaKey || e.ctrlKey) && e.key === "m") { e.preventDefault(); mergeSelected(); }
      if ((e.metaKey || e.ctrlKey) && e.key === "t") { e.preventDefault(); translateAll(); }
      if ((e.metaKey || e.ctrlKey) && e.key === "a" && !(e.target instanceof HTMLInputElement)) { e.preventDefault(); selectAll(); }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [editingCell, deleteSelected, mergeSelected, translateAll, selectAll]);

  const allSelected = subtitles.length > 0 && selectedIds.size === subtitles.length;

  return (
    <div className="flex flex-col h-full bg-surface relative">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-3">
          <h2 className="text-[13px] font-medium text-text-primary">字幕编辑</h2>
          {subtitles.length > 0 && <span className="text-[11px] text-text-muted bg-[rgba(0,0,0,0.04)] px-2 py-0.5 rounded-full">{subtitles.length} 条</span>}
          {selectedIds.size > 0 && <span className="text-[11px] text-accent bg-accent-dim px-2 py-0.5 rounded-full font-medium">已选 {selectedIds.size}</span>}
        </div>
        <div className="flex items-center gap-1.5">
          <label className="px-2.5 py-1.5 text-[12px] rounded-md text-text-secondary hover:text-text-primary hover:bg-surface-hover transition-all border border-border cursor-pointer btn-press">
            导入
            <input type="file" accept=".srt,.vtt,.ass" className="hidden" onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) { filesApi.upload(file).then(({ file_path }) => { useAppStore.getState().setSubtitleFile(file_path); subtitlesApi.load(file_path).then((subFile) => { useAppStore.getState().setSubtitles(subFile.segments); }).catch((err) => { useAppStore.getState().setError(err instanceof Error ? err.message : "Failed to load subtitle file"); useAppStore.getState().setSubtitleFile(null); }); }).catch((err) => { useAppStore.getState().setError(err instanceof Error ? err.message : "Upload failed"); }); }
            }} />
          </label>
          <button disabled={!subtitleFile || subtitles.length === 0} onClick={handleSave}
            className="px-2.5 py-1.5 text-[12px] rounded-md text-text-secondary hover:text-text-primary hover:bg-surface-hover transition-all border border-border disabled:opacity-30 disabled:cursor-not-allowed btn-press">
            保存
          </button>
          <div className="relative">
            <button disabled={!subtitleFile} onClick={(e) => { e.stopPropagation(); setShowExportMenu(!showExportMenu); }}
              className="px-2.5 py-1.5 text-[12px] rounded-md bg-accent-dim text-accent hover:bg-accent/15 transition-all font-medium disabled:opacity-30 disabled:cursor-not-allowed btn-press">
              导出
            </button>
            {showExportMenu && (
              <div onClick={(e) => e.stopPropagation()} className="absolute right-0 top-full mt-1 bg-surface border border-border rounded-xl z-50 p-3 w-56 space-y-3 shadow-md">
                {/* Format */}
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider font-medium block mb-1.5">格式</label>
                  <div className="grid grid-cols-5 gap-1">
                    {(["srt", "vtt", "ass", "txt", "json"] as const).map((fmt) => (
                      <button key={fmt} onClick={() => setExportFormat(fmt)}
                        className={`py-1 rounded text-[11px] font-medium transition-all btn-press ${
                          exportFormat === fmt ? "bg-accent-dim text-accent" : "text-text-muted hover:text-text-secondary hover:bg-[rgba(0,0,0,0.03)]"
                        }`}>
                        {fmt.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                {/* Language mode */}
                <div>
                  <label className="text-[11px] text-text-muted uppercase tracking-wider font-medium block mb-1.5">语言</label>
                  <div className="space-y-0.5">
                    {([
                      { id: "bilingual" as const, label: "中英双语", desc: "原文 + 译文" },
                      { id: "original" as const, label: "仅原文", desc: "原始语言字幕" },
                      { id: "translated" as const, label: "仅译文", desc: "翻译后的字幕" },
                    ]).map((m) => (
                      <button key={m.id} onClick={() => setExportMode(m.id)}
                        className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-left transition-all ${
                          exportMode === m.id ? "bg-accent-dim text-accent" : "text-text-secondary hover:bg-[rgba(0,0,0,0.03)]"
                        }`}>
                        <span className="text-[12px] font-medium">{m.label}</span>
                        <span className="text-[10px] text-text-muted">{m.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                {/* Export button */}
                <button onClick={() => handleExport()}
                  className="w-full py-1.5 rounded-md bg-accent text-white text-[12px] font-medium hover:bg-accent-hover transition-all btn-press shadow-md">
                  导出 {exportFormat.toUpperCase()}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toolbar */}
      {subtitles.length > 0 && (
        <div className="flex items-center gap-0.5 px-3 py-1.5 border-b border-border bg-[rgba(0,0,0,0.01)]">
          <button onClick={allSelected ? deselectAll : selectAll} className="p-1.5 rounded text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all btn-press" title="全选 (Ctrl+A)">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" /></svg>
          </button>
          <div className="w-px h-3.5 bg-border mx-1" />
          <button onClick={deleteSelected} disabled={selectedIds.size === 0} className="p-1.5 rounded text-text-muted hover:text-red-500 hover:bg-red-50 transition-all disabled:opacity-30 btn-press" title="删除选中 (Delete)">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
          </button>
          <button onClick={mergeSelected} disabled={selectedIds.size < 2} className="p-1.5 rounded text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all disabled:opacity-30 btn-press" title="合并选中 (Ctrl+M)">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" /></svg>
          </button>
          <div className="w-px h-3.5 bg-border mx-1" />
          {showTranslateActions && (
            <>
              <button onClick={translateAll} disabled={!subtitleFile || isTranslating} className="px-2 py-1 text-[12px] rounded text-text-muted hover:text-accent hover:bg-accent-dim transition-all disabled:opacity-30 btn-press" title="翻译全部 (Ctrl+T)">
                {isTranslating ? "翻译中..." : "翻译全部"}
              </button>
              <button onClick={retranslateAll} disabled={isTranslating} className="px-2 py-1 text-[12px] rounded text-text-muted hover:text-accent hover:bg-accent-dim transition-all disabled:opacity-30 btn-press">
                重新翻译全部
              </button>
            </>
          )}
        </div>
      )}

      {/* Custom prompt */}
      {showPrompt && config.needOptimize && (
        <div className="border-b border-border">
          <button onClick={() => setPromptExpanded(!promptExpanded)} className="w-full flex items-center gap-2 px-4 py-2 text-[12px] text-text-muted hover:text-text-secondary hover:bg-[rgba(0,0,0,0.02)] transition-all">
            <svg className={`w-3 h-3 transition-transform ${promptExpanded ? "rotate-90" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
            <span>自定义优化提示词</span>
            {config.customPrompt && <span className="text-[10px] text-accent bg-accent-dim px-1.5 py-0.5 rounded-full">已设置</span>}
          </button>
          {promptExpanded && (
            <div className="px-4 pb-3">
              <textarea
                value={config.customPrompt || ""}
                onChange={(e) => {
                  useAppStore.getState().setConfig({ customPrompt: e.target.value });
                  if (promptDebounceRef.current) clearTimeout(promptDebounceRef.current);
                  promptDebounceRef.current = setTimeout(() => { configApi.update("custom_prompt", e.target.value); }, 500);
                }}
                placeholder="输入自定义提示词，用于指导字幕优化和翻译..."
                className="w-full h-28 text-[13px] text-text-primary bg-surface border border-border rounded-lg p-3 resize-none focus:outline-none focus:border-accent transition-colors placeholder:text-text-muted"
              />
              <p className="text-[11px] text-text-muted mt-1.5">提示词将用于指导 LLM 优化和翻译字幕内容</p>
            </div>
          )}
        </div>
      )}

      {/* Context menu */}
      {contextMenu && (
        <div className="fixed bg-surface border border-border rounded-lg z-50 py-1 min-w-[140px] shadow-md" style={{ left: contextMenu.x, top: contextMenu.y }}>
          <button onClick={() => { deleteSelected(); setContextMenu(null); }} className="w-full px-3 py-1.5 text-left text-[12px] text-text-secondary hover:text-red-600 hover:bg-red-50 flex items-center gap-2">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
            删除
          </button>
          <button onClick={() => { mergeSelected(); setContextMenu(null); }} disabled={selectedIds.size < 2} className="w-full px-3 py-1.5 text-left text-[12px] text-text-secondary hover:text-text-primary hover:bg-[rgba(0,0,0,0.03)] flex items-center gap-2 disabled:opacity-30">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" /></svg>
            合并
          </button>
          <div className="h-px bg-border my-1" />
          <button onClick={() => { translateAll(); setContextMenu(null); }} disabled={!subtitleFile} className="w-full px-3 py-1.5 text-left text-[12px] text-accent hover:bg-accent-dim flex items-center gap-2 disabled:opacity-30">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" /></svg>
            翻译全部
          </button>
          <button onClick={() => { if (contextMenu.id) { const sub = subtitles.find((s) => s.id === contextMenu.id); if (sub) startEdit(sub.id, "text", sub.text); } setContextMenu(null); }}
            className="w-full px-3 py-1.5 text-left text-[12px] text-text-secondary hover:text-text-primary hover:bg-[rgba(0,0,0,0.03)] flex items-center gap-2">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
            编辑
          </button>
        </div>
      )}

      {/* Table */}
      <div ref={tableScrollRef} className="flex-1 overflow-auto">
        {subtitles.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent-dim to-[rgba(37,99,235,0.04)] flex items-center justify-center mb-4 border border-accent/10">
              <svg className="w-7 h-7 text-accent/50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
            </div>
            <p className="text-[13px] text-text-secondary mb-1 font-medium">暂无字幕数据</p>
            <p className="text-[12px] text-text-muted">转录完成后字幕将在此显示</p>
          </div>
        ) : (
          <table className="grid w-full text-[13px]">
            <thead className="sticky top-0 z-10 grid">
              <tr className="grid grid-cols-[40px_40px_148px_minmax(0,1fr)_minmax(0,1fr)] bg-surface border-b border-border">
                <th className="text-left px-3 py-2 text-[11px] text-text-muted font-medium w-10"><input type="checkbox" checked={allSelected} onChange={allSelected ? deselectAll : selectAll} className="accent-accent w-3 h-3" /></th>
                <th className="text-left px-3 py-2 text-[11px] text-text-muted font-medium w-10">#</th>
                <th className="text-left px-3 py-2 text-[11px] text-text-muted font-medium w-36">时间</th>
                <th className="text-left px-3 py-2 text-[11px] text-text-muted font-medium">原文</th>
                <th className="text-left px-3 py-2 text-[11px] text-text-muted font-medium">译文</th>
              </tr>
            </thead>
            <tbody
              className="relative grid"
              style={{ height: `${rowVirtualizer.getTotalSize()}px` }}
            >
              {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                const sub = subtitles[virtualRow.index];
                const idx = virtualRow.index;
                return (
                <tr
                  key={sub.id}
                  data-index={virtualRow.index}
                  ref={(row) => {
                    if (row) {
                      rowRefs.current.set(sub.id, row);
                      rowVirtualizer.measureElement(row);
                    } else {
                      rowRefs.current.delete(sub.id);
                    }
                  }}
                  tabIndex={-1}
                  className={`subtitle-row group absolute left-0 top-0 grid w-full grid-cols-[40px_40px_148px_minmax(0,1fr)_minmax(0,1fr)] cursor-pointer transition-[background-color,box-shadow] duration-300 focus:outline-none ${selectedIds.has(sub.id) ? "selected" : ""} ${
                    focusedRowId === sub.id
                      ? "bg-amber-50 shadow-[inset_3px_0_0_#d97706]"
                      : idx % 2 === 0
                        ? ""
                        : "bg-[rgba(0,0,0,0.015)]"
                  }`}
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                  onClick={() => toggleSelect(sub.id)} onContextMenu={(e) => handleContextMenu(e, sub.id)}>
                  <td className="px-3 py-2 border-b border-[rgba(0,0,0,0.04)]">
                    <input type="checkbox" checked={selectedIds.has(sub.id)} onChange={() => toggleSelect(sub.id)} onClick={(e) => e.stopPropagation()} className="accent-accent w-3 h-3" />
                  </td>
                  <td className="px-3 py-2 text-[12px] text-text-muted font-mono border-b border-[rgba(0,0,0,0.04)]">{sub.id}</td>
                  <td className="min-w-0 overflow-hidden px-3 py-2 border-b border-[rgba(0,0,0,0.04)]">
                    <div className="flex flex-col gap-0.5 leading-tight">
                      <span className="text-[12px] text-text-muted font-mono">{sub.start}</span>
                      <span className="text-[12px] text-text-muted font-mono">&rarr; {sub.end}</span>
                    </div>
                  </td>
                  <td className="min-w-0 overflow-hidden px-3 py-2 border-b border-[rgba(0,0,0,0.04)]" onDoubleClick={(e) => { e.stopPropagation(); startEdit(sub.id, "text", sub.text); }}>
                    {editingCell?.id === sub.id && editingCell?.field === "text" ? (
                      <input autoFocus className="inline-edit text-text-primary" value={editValue} onChange={(e) => setEditValue(e.target.value)} onBlur={commitEdit} onKeyDown={(e) => { if (e.key === "Enter") commitEdit(); if (e.key === "Escape") setEditingCell(null); }} onClick={(e) => e.stopPropagation()} />
                    ) : (
                      <span className="flex items-start gap-2 text-text-primary group-hover:text-text-primary transition-colors">
                        {sub.speaker && (
                          <span className="mt-0.5 shrink-0 rounded bg-accent-dim px-1.5 py-0.5 font-mono text-[10px] font-semibold text-accent" title={sub.speaker}>
                            {formatSpeakerLabel(sub.speaker)}
                          </span>
                        )}
                        <span className="min-w-0 break-words">{sub.text}</span>
                      </span>
                    )}
                  </td>
                  <td className="min-w-0 overflow-hidden px-3 py-2 border-b border-[rgba(0,0,0,0.04)]" onDoubleClick={(e) => { e.stopPropagation(); startEdit(sub.id, "translated", sub.translated); }}>
                    {editingCell?.id === sub.id && editingCell?.field === "translated" ? (
                      <input autoFocus className="inline-edit text-accent" value={editValue} onChange={(e) => setEditValue(e.target.value)} onBlur={commitEdit} onKeyDown={(e) => { if (e.key === "Enter") commitEdit(); if (e.key === "Escape") setEditingCell(null); }} onClick={(e) => e.stopPropagation()} />
                    ) : (
                      <span className="block min-w-0 break-words text-text-muted group-hover:text-accent transition-colors">
                        {sub.translated.trim() || <span className="text-amber-700 italic">待翻译</span>}
                      </span>
                    )}
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Status bar */}
      {subtitles.length > 0 && (
        <div className="flex items-center justify-between px-4 py-1.5 border-t border-border bg-[rgba(0,0,0,0.015)]">
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-text-muted">{subtitles.length} 条字幕</span>
            <span className="text-[11px] text-text-muted">{subtitles.filter((s) => s.translated.trim()).length} 已翻译</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-[11px] text-text-muted">Delete 删除 · Ctrl+M 合并 · Ctrl+T 翻译 · 双击时间跳转</span>
            <div className="flex items-center gap-2">
              {subtitles.every((s) => s.translated.trim()) ? (
                <><span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /><span className="text-[11px] text-emerald-600">翻译完成</span></>
              ) : (
                <><span className="w-1.5 h-1.5 rounded-full bg-amber-400" /><span className="text-[11px] text-text-muted">待翻译</span></>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatSpeakerLabel(speaker: string): string {
  const match = speaker.match(/(\d+)\s*$/);
  return match ? `S${match[1]}` : speaker;
}
