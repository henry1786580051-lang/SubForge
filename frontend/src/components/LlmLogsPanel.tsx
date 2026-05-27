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
    <div className="flex flex-col h-full bg-[var(--background)] overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-[var(--border)] bg-[var(--surface)]">
        <h2 className="text-sm font-medium text-[var(--text-primary)]">LLM 请求日志</h2>
        <div className="flex items-center gap-2">
          <button onClick={handleClear}
            className="px-2.5 py-1 text-[11px] rounded-md text-[var(--text-muted)] hover:text-red-500 hover:bg-red-50 transition-all border border-[var(--border)]">
            清除日志
          </button>
          <button onClick={() => setActiveView("workflow")} className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[rgba(0,0,0,0.04)] transition-all">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="px-5 py-3 border-b border-[var(--border)]">
        <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索任务ID、文件名、模型、阶段..." className="input-field" />
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Table */}
        <div className={`${selected ? "w-1/2" : "flex-1"} overflow-auto transition-all`}>
          {loading ? (
            <div className="flex items-center justify-center h-40"><div className="w-6 h-6 rounded-full border-2 border-[var(--accent)]/20 border-t-[var(--accent)] animate-spin" /></div>
          ) : logs.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-[12px] text-[var(--text-muted)]">暂无日志记录</div>
          ) : (
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-[var(--surface)]">
                <tr className="border-b border-[var(--border)] text-[var(--text-muted)]">
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
                    className={`border-b border-[var(--border)] cursor-pointer hover:bg-[rgba(0,0,0,0.02)] transition-colors ${
                      selected === log ? "bg-[var(--accent-dim)]" : ""
                    }`}>
                    <td className="px-3 py-2 font-mono text-[10px] text-[var(--text-muted)]">{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : "-"}</td>
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
            <div className="flex items-center justify-center gap-2 py-3 border-t border-[var(--border)]">
              <button onClick={() => fetchLogs(page - 1, search)} disabled={page <= 1}
                className="px-2 py-1 text-[11px] rounded border border-[var(--border)] disabled:opacity-30">上一页</button>
              <span className="text-[11px] text-[var(--text-muted)]">{page} / {pages} ({total}条)</span>
              <button onClick={() => fetchLogs(page + 1, search)} disabled={page >= pages}
                className="px-2 py-1 text-[11px] rounded border border-[var(--border)] disabled:opacity-30">下一页</button>
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="w-1/2 border-l border-[var(--border)] overflow-auto p-4 space-y-3 bg-[var(--surface)]">
            <div className="flex items-center justify-between">
              <h3 className="text-[12px] font-medium text-[var(--text-primary)]">请求详情</h3>
              <button onClick={() => setSelected(null)} className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--text-secondary)]">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
            <div className="space-y-1.5 text-[11px]">
              <div className="flex justify-between"><span className="text-[var(--text-muted)]">任务ID</span><span className="font-mono">{selected.task_id || "-"}</span></div>
              <div className="flex justify-between"><span className="text-[var(--text-muted)]">文件</span><span className="truncate ml-2">{selected.file_name || "-"}</span></div>
              <div className="flex justify-between"><span className="text-[var(--text-muted)]">阶段</span><span>{selected.stage || "-"}</span></div>
              <div className="flex justify-between"><span className="text-[var(--text-muted)]">模型</span><span className="font-mono">{selected.model || "-"}</span></div>
              <div className="flex justify-between"><span className="text-[var(--text-muted)]">耗时</span><span className="font-mono">{fmtDuration(selected.duration_ms)}</span></div>
            </div>
            {!!selected.request && (
              <div>
                <h4 className="text-[11px] font-medium text-[var(--text-muted)] mb-1">请求内容</h4>
                <pre className="text-[10px] bg-[rgba(0,0,0,0.03)] rounded-lg p-3 overflow-auto max-h-60 font-mono whitespace-pre-wrap break-all">{String(JSON.stringify(selected.request, null, 2))}</pre>
              </div>
            )}
            {!!selected.response && (
              <div>
                <h4 className="text-[11px] font-medium text-[var(--text-muted)] mb-1">响应内容</h4>
                <pre className="text-[10px] bg-[rgba(0,0,0,0.03)] rounded-lg p-3 overflow-auto max-h-60 font-mono whitespace-pre-wrap break-all">{String(JSON.stringify(selected.response, null, 2))}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
