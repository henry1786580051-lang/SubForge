"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useAppStore } from "@/store/appStore";
import { llmLogsApi, type LlmLogEntry } from "@/lib/api";

export function LlmLogsPanel() {
  const { setActiveView } = useAppStore();
  const [logs, setLogs] = useState<LlmLogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<LlmLogEntry | null>(null);

  const fetchLogs = useCallback(async (p: number, q: string) => {
    setLoading(true);
    try {
      const data = await llmLogsApi.list(p, q);
      setLogs(data.logs);
      setTotal(data.total);
      setPages(data.pages);
      setPage(data.page);
    } catch { /* ignore */ }
    finally { setLoading(false); }
  }, []);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => { fetchLogs(1, search); }, 300);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchLogs, search]);

  const handleClear = async () => {
    await llmLogsApi.clear();
    fetchLogs(1, "");
  };

  const fmtDuration = (ms?: number) => {
    if (!ms) return "-";
    return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  };

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-surface">
        <h2 className="text-[13px] font-medium text-text-primary">LLM 请求日志</h2>
        <div className="flex items-center gap-2">
          <button onClick={handleClear}
            className="px-2.5 py-1 text-[12px] rounded-md text-text-muted hover:text-red-500 hover:bg-red-50 transition-all border border-border btn-press">
            清除日志
          </button>
          <button onClick={() => setActiveView("workflow")} className="p-1.5 rounded-md text-text-muted hover:text-text-secondary hover:bg-surface-hover transition-all btn-press">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="px-5 py-3 border-b border-border">
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索任务ID、文件名、模型、阶段..." className="input-field" />
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Table */}
        <div className={`${selected ? "w-1/2" : "flex-1"} overflow-auto transition-all`}>
          {loading ? (
            <div className="flex items-center justify-center h-40"><div className="w-6 h-6 rounded-full border-2 border-accent/20 border-t-accent animate-spin" /></div>
          ) : logs.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-[13px] text-text-muted">暂无日志记录</div>
          ) : (
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-border text-text-muted">
                  <th className="px-3 py-2 text-left font-medium">时间</th>
                  <th className="px-3 py-2 text-left font-medium">文件</th>
                  <th className="px-3 py-2 text-left font-medium">阶段</th>
                  <th className="px-3 py-2 text-left font-medium">模型</th>
                  <th className="px-3 py-2 text-right font-medium">耗时</th>
                  <th className="px-3 py-2 text-right font-medium">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, i) => (
                  <tr key={i} onClick={() => setSelected(log)}
                    className={`border-b border-border cursor-pointer hover:bg-[rgba(0,0,0,0.02)] transition-colors ${
                      selected === log ? "bg-accent-dim" : ""
                    }`}>
                    <td className="px-3 py-2 font-mono text-[10px] text-text-muted">{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : "-"}</td>
                    <td className="px-3 py-2 truncate max-w-[120px]">{log.file_name || "-"}</td>
                    <td className="px-3 py-2"><span className="px-1.5 py-0.5 rounded bg-[rgba(0,0,0,0.04)] text-[10px]">{log.stage || "-"}</span></td>
                    <td className="px-3 py-2 font-mono text-[10px]">{log.model || "-"}</td>
                    <td className="px-3 py-2 text-right font-mono">{fmtDuration(log.duration_ms)}</td>
                    <td className="px-3 py-2 text-right font-mono">{log.tokens ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Pagination */}
          {pages > 1 && (
            <div className="flex items-center justify-center gap-2 py-3 border-t border-border">
              <button onClick={() => fetchLogs(page - 1, search)} disabled={page <= 1}
                className="px-2 py-1 text-[12px] rounded border border-border disabled:opacity-30 btn-press">上一页</button>
              <span className="text-[12px] text-text-muted">{page} / {pages} ({total}条)</span>
              <button onClick={() => fetchLogs(page + 1, search)} disabled={page >= pages}
                className="px-2 py-1 text-[12px] rounded border border-border disabled:opacity-30 btn-press">下一页</button>
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="w-1/2 border-l border-border overflow-auto p-4 space-y-3 bg-surface">
            <div className="flex items-center justify-between">
              <h3 className="text-[13px] font-medium text-text-primary">请求详情</h3>
              <button onClick={() => setSelected(null)} className="p-1 rounded text-text-muted hover:text-text-secondary">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
            <div className="space-y-1.5 text-[12px]">
              <div className="flex justify-between"><span className="text-text-muted">任务ID</span><span className="font-mono">{selected.task_id || "-"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">文件</span><span className="truncate ml-2">{selected.file_name || "-"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">阶段</span><span>{selected.stage || "-"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">模型</span><span className="font-mono">{selected.model || "-"}</span></div>
              <div className="flex justify-between"><span className="text-text-muted">耗时</span><span className="font-mono">{fmtDuration(selected.duration_ms)}</span></div>
            </div>
            {!!selected.request && (
              <div>
                <h4 className="text-[12px] font-medium text-text-muted mb-1">请求内容</h4>
                <pre className="text-[11px] bg-[rgba(0,0,0,0.03)] rounded-lg p-3 overflow-auto max-h-60 font-mono whitespace-pre-wrap break-all">{String(JSON.stringify(selected.request, null, 2))}</pre>
              </div>
            )}
            {!!selected.response && (
              <div>
                <h4 className="text-[12px] font-medium text-text-muted mb-1">响应内容</h4>
                <pre className="text-[11px] bg-[rgba(0,0,0,0.03)] rounded-lg p-3 overflow-auto max-h-60 font-mono whitespace-pre-wrap break-all">{String(JSON.stringify(selected.response, null, 2))}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
